"""
Plot NeutroHydro results from result folders (results_lowrank and results_robustpca).

Generates and saves the following plots per folder:
 - NVIP stacked bar (VIP_T, VIP_I, VIP_F)
 - NVIP agg bar
 - pi_G bar (baseline fraction per ion)
 - G histogram (sample-level baseline fraction)
 - Mineral fraction heatmap (if mineral_inversion.csv present)

Usage:
    python examples/plot_results.py results_lowrank
    python examples/plot_results.py results_robustpca

"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')


def plot_nvip_stack(df_nvip, outpath):
    df = df_nvip.copy()
    df = df.sort_values('VIP_agg', ascending=False)
    inds = np.arange(len(df))

    plt.figure(figsize=(8, 5))
    plt.bar(inds, df['VIP_T'], label='VIP_T')
    plt.bar(inds, df['VIP_I'], bottom=df['VIP_T'], label='VIP_I')
    plt.bar(inds, df['VIP_F'], bottom=df['VIP_T'] + df['VIP_I'], label='VIP_F')
    plt.xticks(inds, df['feature'], rotation=45, ha='right')
    plt.ylabel('VIP (stacked)')
    plt.title('NVIP - Channel decomposition (stacked)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_nvip_agg(df_nvip, outpath):
    df = df_nvip.copy()
    df = df.sort_values('VIP_agg', ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x='feature', y='VIP_agg', data=df, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('VIP_agg')
    plt.title('NVIP aggregated importance')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_piG(df_nsr, outpath):
    df = df_nsr.copy()
    df = df.sort_values('pi_G', ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x='ion', y='pi_G', data=df, palette='magma')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('pi_G (baseline fraction)')
    plt.title('Baseline fraction per ion (pi_G)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_G_hist(df_samples, outpath):
    plt.figure(figsize=(6, 4))
    sns.histplot(df_samples['G'], bins=20, kde=False, color='steelblue')
    plt.xlabel('Sample G (baseline fraction)')
    plt.title('Distribution of sample-level baseline fraction G')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_mineral_heatmap(mineral_df, outpath):
    # Select fraction columns
    frac_cols = [c for c in mineral_df.columns if c.endswith('_frac')]
    if not frac_cols:
        return
    data = mineral_df[frac_cols]
    # Convert to matrix (samples x minerals)
    plt.figure(figsize=(10, max(4, 0.25 * data.shape[0])))
    sns.heatmap(data.T, cmap='viridis', cbar_kws={'label': 'Mineral fraction'})
    plt.yticks(np.arange(len(frac_cols)) + 0.5, [c.replace('_frac', '') for c in frac_cols], rotation=0)
    plt.xlabel('Sample index')
    plt.title('Mineral fractions (per sample)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def run_folder(folder):
    # Read CSVs
    nvip_path = os.path.join(folder, 'nvip.csv')
    nsr_path = os.path.join(folder, 'nsr.csv')
    samples_path = os.path.join(folder, 'samples.csv')
    mineral_path = os.path.join(folder, 'mineral_inversion.csv')

    if not os.path.exists(nvip_path) or not os.path.exists(nsr_path) or not os.path.exists(samples_path):
        print(f"One or more required CSVs missing in {folder}")
        return

    df_nvip = pd.read_csv(nvip_path)
    df_nsr = pd.read_csv(nsr_path)
    df_samples = pd.read_csv(samples_path)

    # Plots
    os.makedirs(folder, exist_ok=True)
    plot_nvip_stack(df_nvip, os.path.join(folder, 'nvip_stack.png'))
    plot_nvip_agg(df_nvip, os.path.join(folder, 'nvip_agg.png'))
    plot_piG(df_nsr, os.path.join(folder, 'piG.png'))
    plot_G_hist(df_samples, os.path.join(folder, 'G_hist.png'))

    if os.path.exists(mineral_path):
        df_mineral = pd.read_csv(mineral_path)
        plot_mineral_heatmap(df_mineral, os.path.join(folder, 'mineral_heatmap.png'))
        print(f"Saved plots to {folder}: nvip_stack.png, nvip_agg.png, piG.png, G_hist.png, mineral_heatmap.png")
    else:
        print(f"Saved plots to {folder}: nvip_stack.png, nvip_agg.png, piG.png, G_hist.png")


if __name__ == '__main__':
    folders = sys.argv[1:] or ['results_lowrank', 'results_robustpca']
    for f in folders:
        run_folder(f)
