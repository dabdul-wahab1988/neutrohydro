import os
import sys
import argparse
import pandas as pd
import numpy as np
import questionary
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path

from neutrohydro.pipeline import NeutroHydroPipeline, PipelineConfig
from neutrohydro.interpretation import GeochemicalInterpreter
from neutrohydro.quality_check import assess_water_quality, check_sanity
from neutrohydro.visualization import (
    generate_report, 
    PLOT_REGISTRY, 
    PRESETS,
    create_figure, 
    mg_to_meq, 
    calculate_ion_sums
)
from neutrohydro.minerals import ION_MASSES, ION_CHARGES

console = Console()

class NeutroHydroCLI:
    def __init__(self):
        self.df = None
        self.pipeline = None
        self.results = None
        self.feature_names = ["Ca2+", "Mg2+", "Na+", "K+", "HCO3-", "Cl-", "SO42-", "NO3-", "F-"]
        self.ion_mapping = {
            "Ca": "Ca2+", "Mg": "Mg2+", "Na": "Na+", "K": "K+", 
            "HCO3": "HCO3-", "Cl": "Cl-", "SO4": "SO42-",
            "NO3": "NO3-", "F": "F-"
        }
        # Default Configuration
        self.config_params = {
            "n_components": 5,
            "baseline_type": "robust_pca",
            "baseline_rank": 2,
            "run_mineral_inference": True,
            "run_thermodynamic_validation": False,
            "log_transform_predictors": False,
            "log_transform_target": True,
            "target_column": None
        }

    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_welcome(self):
        self.clear()
        console.print(Panel.fit(
            "[bold blue]NEUTROHYDRO[/]\n[italic]Advanced Hydrogeochemical Analysis Wizard[/]",
            border_style="blue"
        ))

    def load_data(self):
        path = questionary.path("Select your CSV data file:").ask()
        if not path or not os.path.exists(path):
            console.print("[red]Error: Invalid file path.[/]")
            return False

        try:
            self.df = pd.read_csv(path)
            console.print(f"[green][OK] Loaded {len(self.df)} samples.[/]")
            
            # Preview columns
            table = Table(title="Data Columns")
            table.add_column("Column Name")
            table.add_column("Mapped To", style="cyan")
            
            found_ions = []
            rename_dict = {}
            
            for col in self.df.columns:
                mapped = None
                for key, val in self.ion_mapping.items():
                    if key.lower() in col.lower():
                        mapped = val
                        found_ions.append(val)
                        rename_dict[col] = key # Map to short name first (e.g. 'Ca')
                        break
                table.add_row(col, mapped or "-")
            
            console.print(table)
            
            confirm = questionary.confirm("Do you want to proceed with this mapping?").ask()
            if not confirm:
                console.print("[yellow]Please rename your CSV columns to match (Ca, Mg, Na, K, HCO3, Cl, SO4) and try again.[/]")
                return False
            
            self.df = self.df.rename(columns=rename_dict)
            
            # Update feature names based on what was found
            # We map back to full names (e.g. 'Ca' -> 'Ca2+')
            self.feature_names = []
            for short_name in rename_dict.values():
                if short_name in self.ion_mapping:
                    self.feature_names.append(self.ion_mapping[short_name])
            
            # Ensure we have the minimum required ions
            required = ["Ca2+", "Mg2+", "Na+", "HCO3-", "Cl-", "SO42-"]
            missing = [ion for ion in required if ion not in self.feature_names]
            
            if missing:
                console.print(f"[red]Warning: Missing critical ions for analysis: {', '.join(missing)}[/]")
                if not questionary.confirm("Proceed anyway? (Analysis may be limited)").ask():
                    return False
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error loading CSV: {e}[/]")
            return False

    def quality_check(self):
        if self.df is None:
            console.print("[red]No data loaded.[/]")
            return

        console.print("[bold cyan]Running Data Quality Checks...[/]")
        
        # 1. Sanity Check
        # We need to map columns back to standard names for the check
        # The df currently has short names (Ca, Mg) but check_sanity expects standard names or we pass mapping
        # Actually check_sanity expects a DataFrame. Let's see what it expects.
        # It expects columns like 'Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4'
        # Our df has these short names now.
        
        sanity_report = check_sanity(self.df)
        
        console.print("\n[bold]Sanity Check Report:[/]")
        if sanity_report['critical_failures']:
            console.print(f"[red]CRITICAL FAILURES: {len(sanity_report['critical_failures'])} samples rejected.[/]")
            console.print("Reasons: " + ", ".join(set(sanity_report['failure_reasons'])))
            if not questionary.confirm("Continue despite critical failures?").ask():
                return
        else:
            console.print("[green]Sanity Check Passed.[/]")
            
        console.print(f"Mean CBE: {sanity_report['mean_cbe']:.2f}%")
        
        # 2. WHO Assessment
        console.print("\n[bold]WHO Water Quality Assessment:[/]")
        quality_df = assess_water_quality(self.df)
        
        # Summary of exceedances
        exceedance_cols = [c for c in quality_df.columns if "_Exceeds" in c]
        if exceedance_cols:
            table = Table(title="WHO Exceedances")
            table.add_column("Parameter")
            table.add_column("Samples Exceeding")
            
            for col in exceedance_cols:
                count = quality_df[col].sum()
                if count > 0:
                    param = col.replace("_Exceeds", "")
                    table.add_row(param, str(count))
            console.print(table)
            
        # Summary of Inferred Sources
        if "Inferred_Source" in quality_df.columns:
            console.print("\n[bold]Inferred Pollution Sources:[/]")
            sources = quality_df["Inferred_Source"].value_counts()
            for source, count in sources.items():
                console.print(f"• {source}: {count} samples")

    def configure_pipeline(self):
        """Advanced configuration menu."""
        console.print("[bold cyan]Pipeline Configuration[/]")
        
        # 1. Target Variable
        if self.df is not None:
            cols = self.df.columns.tolist()
            default_target = "TDS" if "TDS" in cols else cols[-1]
            self.config_params["target_column"] = questionary.select(
                "Select Target Variable (e.g., TDS, EC):",
                choices=cols,
                default=default_target
            ).ask()
        
        # 2. Baseline Strategy
        self.config_params["baseline_type"] = questionary.select(
            "Select Baseline Strategy:",
            choices=[
                questionary.Choice("Robust PCA (Best for anomalies)", "robust_pca"),
                questionary.Choice("Low Rank (SVD)", "low_rank")
            ],
            default=self.config_params["baseline_type"]
        ).ask()
        
        # 3. Mineral Inversion
        self.config_params["run_mineral_inference"] = questionary.confirm(
            "Run Mineral Stoichiometric Inversion?",
            default=self.config_params["run_mineral_inference"]
        ).ask()
        
        if self.config_params["run_mineral_inference"]:
            self.config_params["run_thermodynamic_validation"] = questionary.confirm(
                "Use PHREEQC Thermodynamic Validation? (Requires pH, Temp)",
                default=self.config_params["run_thermodynamic_validation"]
            ).ask()
            
        # 4. Advanced Params (Optional)
        if questionary.confirm("Show advanced parameters (n_components, logs)?", default=False).ask():
            self.config_params["n_components"] = int(questionary.text(
                "Number of PLS Components:", 
                default=str(self.config_params["n_components"])
            ).ask())
            
            self.config_params["log_transform_predictors"] = questionary.confirm(
                "Log-transform predictors?",
                default=self.config_params["log_transform_predictors"]
            ).ask()
            
            self.config_params["log_transform_target"] = questionary.confirm(
                "Log-transform target?",
                default=self.config_params["log_transform_target"]
            ).ask()

        console.print("[green]Configuration updated.[/]")

    def run_pipeline(self):
        if self.df is None: 
            console.print("[red]No data loaded.[/]")
            return
        
        # Ensure configuration is set (at least target)
        if self.config_params["target_column"] is None:
            self.configure_pipeline()
            
        target_col = self.config_params["target_column"]
        if target_col not in self.df.columns:
            console.print(f"[red]Target column '{target_col}' not found in data. Please re-configure.[/]")
            self.configure_pipeline()
            target_col = self.config_params["target_column"]

        config = PipelineConfig(
            log_transform=self.config_params["log_transform_predictors"],
            baseline_type=self.config_params["baseline_type"],
            baseline_rank=self.config_params["baseline_rank"],
            run_mineral_inference=self.config_params["run_mineral_inference"],
            run_thermodynamic_validation=self.config_params["run_thermodynamic_validation"],
            n_components=self.config_params["n_components"]
        )
        
        self.pipeline = NeutroHydroPipeline(config)
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            progress.add_task(description="Running PNPLS Pipeline...", total=None)
            
            # Prepare X (Features)
            # We use the keys from ion_mapping that exist in the dataframe
            # The dataframe was renamed in load_data to use short keys (Ca, Mg, etc.)
            feature_cols = [col for col in self.df.columns if col in self.ion_mapping]
            X = self.df[feature_cols].values
            
            # Prepare c_meq (Concentrations in meq/L) for Mineral Inversion
            # CRITICAL: Pipeline expects meq/L for stoichiometry, but X is mg/L
            df_meq = mg_to_meq(self.df)
            c_meq_list = []
            for col in feature_cols:
                meq_col = f"{col}_meq"
                if meq_col in df_meq.columns:
                    c_meq_list.append(df_meq[meq_col].values)
                else:
                    # Fallback for non-ionic species or missing factors (should not happen with standard ions)
                    c_meq_list.append(self.df[col].values)
            c_meq = np.column_stack(c_meq_list)
            
            # Prepare y (Target)
            y = self.df[target_col].values
            if self.config_params["log_transform_target"]:
                y = np.log(np.maximum(y, 1.0))
            
            # Prepare Optional Inputs
            pH = self.df['pH'].values if 'pH' in self.df.columns else None
            temp = self.df['Temp'].values if 'Temp' in self.df.columns else 25.0
            Eh = self.df['Eh'].values if 'Eh' in self.df.columns else None
            
            self.results = self.pipeline.fit(
                X, y, 
                feature_names=self.feature_names,
                c_meq=c_meq,
                pH=pH,
                temp=temp,
                Eh=Eh
            )
        
        console.print(f"[green][OK] Pipeline complete. R² Score: {self.results.r2_train:.4f}[/]")

    def wizard_mode(self):
        """Guided step-by-step workflow."""
        self.show_welcome()
        console.print("[bold yellow]Starting Wizard Mode...[/]")
        
        if not self.load_data(): return
        self.quality_check()
        self.configure_pipeline()
        self.run_pipeline()
        self.generate_interpretation()
        
        if questionary.confirm("Generate Visualizations?").ask():
            self.visualization_menu()
            
        if questionary.confirm("Export Results?").ask():
            self.export_results()
            
        console.print("[bold green]Wizard completed successfully![/]")

    def visualization_menu(self):
        if self.results is None:
            console.print("[yellow]Please run the pipeline first.[/]")
            return

        choices = []
        
        # 1. Presets (Panel Plots)
        choices.append(questionary.Separator("--- Multi-Panel Presets ---"))
        for k, v in PRESETS.items():
            choices.append(questionary.Choice(f"Preset: {k} ({v})", k))
            
        # 2. Individual Plots
        choices.append(questionary.Separator("--- Individual Plots ---"))
        for k, v in PLOT_REGISTRY.items():
            choices.append(questionary.Choice(f"{k}: {v['name']}", k))
            
        # 3. Custom
        choices.append(questionary.Separator("--- Advanced ---"))
        choices.append(questionary.Choice("Custom Layout String...", "custom"))
        
        selected = questionary.checkbox(
            "Select plots to generate:",
            choices=choices
        ).ask()
        
        if not selected: return
        
        # Handle Custom Input
        custom_layouts = []
        if "custom" in selected:
            selected.remove("custom")
            console.print("[cyan]Enter layout string (e.g., '[g1|g2][v1]')[/]")
            console.print("Codes: " + ", ".join(PLOT_REGISTRY.keys()))
            layout = questionary.text("Layout:").ask()
            if layout:
                custom_layouts.append(layout)
        
        output_dir = questionary.text("Enter output directory for plots:", default="results/plots").ask()
        os.makedirs(output_dir, exist_ok=True)
        
        data_bundle = {
            'df': self.df,
            'nvip_result': self.results.nvip,
            'nsr_result': self.results.nsr,
            'mineral_result': self.results.mineral_result,
            'samples_result': self.results.sample_attribution
        }
        
        total_tasks = len(selected) + len(custom_layouts)
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task(description="Generating plots...", total=total_tasks)
            
            # Process Standard Selections (Presets & Individual)
            for code in selected:
                # Determine filename and layout
                if code in PRESETS:
                    name = f"panel_{code}"
                    layout = code # create_figure handles preset names
                else:
                    # Individual plot
                    name = PLOT_REGISTRY[code]['name'].lower().replace(" ", "_")
                    layout = f"[{code}]"
                
                path = os.path.join(output_dir, f"{name}.png")
                
                # Special handling for i1, g1, g2 is ONLY needed if they are run individually
                if code == 'i1':
                    from neutrohydro.visualization import plot_ilr_classification
                    plot_ilr_classification(self.df, output_path=path)
                elif code == 'g1':
                    # Gibbs Cations
                    from neutrohydro.visualization import _plot_gibbs_cations, mg_to_meq, apply_publication_style
                    apply_publication_style()
                    df_meq = mg_to_meq(self.df)
                    fig, ax = plt.subplots(figsize=(7, 6))
                    _plot_gibbs_cations(ax, self.df, df_meq)
                    plt.tight_layout()
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                elif code == 'g2':
                    # Gibbs Anions
                    from neutrohydro.visualization import _plot_gibbs_anions, mg_to_meq, apply_publication_style
                    apply_publication_style()
                    df_meq = mg_to_meq(self.df)
                    fig, ax = plt.subplots(figsize=(7, 6))
                    _plot_gibbs_anions(ax, self.df, df_meq)
                    plt.tight_layout()
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    create_figure(layout, data_bundle, output_path=path)
                
                progress.update(task, advance=1)
                
            # Process Custom Layouts
            for i, layout in enumerate(custom_layouts):
                path = os.path.join(output_dir, f"custom_panel_{i+1}.png")
                create_figure(layout, data_bundle, output_path=path)
                progress.update(task, advance=1)
        
        console.print(f"[green][OK] Plots saved to {output_dir}/[/]")

    def generate_interpretation(self):
        if self.results is None:
            console.print("[yellow]Please run the pipeline first.[/]")
            return

        console.print("[bold cyan]Generating Geochemical Interpretation...[/]")
        
        # Ask for Geological Context
        geology = questionary.select(
            "Select the dominant Geological Context for the study area:",
            choices=[
                "Granitic/Gneissic (High F potential)",
                "Basaltic/Volcanic (Hard Rock)",
                "Sedimentary - Carbonate (Limestone/Dolomite/Karst)",
                "Sedimentary - Clastic (Sandstone/Shale)",
                "Metamorphic (Schist/Slate/Quartzite)",
                "Coastal/Marine (Saline Intrusion Risk)",
                "Alluvial/Unconsolidated (Valley Fill)",
                "Anthropogenic/Urban (Contamination Risk)",
                "Unknown/Mixed"
            ]
        ).ask()

        # Get sample codes if available
        sample_codes = self.df['Code'].tolist() if 'Code' in self.df.columns else None
        
        interpreter = GeochemicalInterpreter(self.results, sample_codes=sample_codes, geology=geology)
        report = interpreter.interpret()
        
        # Display Insights
        if report.insights:
            console.print("\n[bold underline]Key Geochemical Insights:[/]")
            for insight in report.insights:
                color = "red" if insight.category == "Paradox" else "green" if insight.category == "Process" else "yellow"
                console.print(f"[{color}]• [{insight.category}] {insight.message}[/]")
                if insight.samples:
                    console.print(f"  [dim]Affected Samples: {', '.join(insight.samples[:5])}{'...' if len(insight.samples)>5 else ''}[/]")
        else:
            console.print("[dim]No specific anomalies or dominant processes detected.[/]")

        # Display Fluoride Summary if relevant
        if not report.fluoride_source_summary.empty:
            console.print("\n[bold underline]Fluoride Source Attribution:[/]")
            summary = report.fluoride_source_summary['Status'].value_counts()
            for status, count in summary.items():
                console.print(f"• {status}: {count} samples")
                
            # Save detailed report
            output_dir = "results/interpretation"
            os.makedirs(output_dir, exist_ok=True)
            report.fluoride_source_summary.to_csv(os.path.join(output_dir, "fluoride_sources.csv"), index=False)
            report.mineral_paradoxes.to_csv(os.path.join(output_dir, "mineral_paradoxes.csv"), index=False)
            console.print(f"\n[green]Detailed interpretation reports saved to {output_dir}/[/]")

    def export_results(self):
        if self.results is None:
            console.print("[yellow]Please run the pipeline first.[/]")
            return
            
        output_dir = questionary.text("Enter output directory for results:", default="results/data").ask()
        self.pipeline.save_results(output_dir)
        console.print(f"[green][OK] Results saved to {output_dir}/[/]")

    def run(self):
        self.show_welcome()
        
        while True:
            choice = questionary.select(
                "Main Menu:",
                choices=[
                    "1. Wizard Mode (Guided Workflow)",
                    "2. Load Data (CSV)",
                    "3. Data Quality Check",
                    "4. Configure Pipeline",
                    "5. Run Full Pipeline",
                    "6. Generate Visualizations",
                    "7. Generate Interpretation Report",
                    "8. Export Results",
                    "9. Exit"
                ]
            ).ask()
            
            if not choice or "Exit" in choice:
                break
            
            if "Wizard" in choice:
                self.wizard_mode()
            elif "Load Data" in choice:
                self.load_data()
            elif "Quality" in choice:
                self.quality_check()
            elif "Configure" in choice:
                self.configure_pipeline()
            elif "Pipeline" in choice:
                self.run_pipeline()
            elif "Visualization" in choice:
                self.visualization_menu()
            elif "Interpretation" in choice:
                self.generate_interpretation()
            elif "Export" in choice:
                self.export_results()
            
            input("\nPress Enter to return to menu...")
            self.show_welcome()

if __name__ == "__main__":
    # Simple check for arguments to decide mode
    if len(sys.argv) > 1:
        # If arguments are provided, use the installed CLI logic (src/neutrohydro/cli.py)
        # This avoids duplicating the argparse logic here.
        from neutrohydro.cli import main
        sys.exit(main())
    else:
        # Interactive Wizard Mode
        cli = NeutroHydroCLI()
        cli.run()
