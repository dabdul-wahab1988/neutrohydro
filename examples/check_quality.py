"""
Example script to check water quality against WHO guidelines using NeutroHydro.
"""
import pandas as pd
import sys
import os

# Ensure we can import neutrohydro
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from neutrohydro.quality_check import add_quality_flags

def main():
    # Load Data
    data_path = "data3.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print("Loaded data:")
    print(df.head())

    # Run Quality Assessment
    print("\nRunning WHO Quality Assessment...")
    df_quality = add_quality_flags(df)

    # Display Results
    print("\nAssessment Results:")
    cols = ['Code', 'Exceedances', 'Inferred_Sources']
    print(df_quality[cols].head())

    # Save to CSV
    output_path = "data3_quality_report.csv"
    df_quality.to_csv(output_path, index=False)
    print(f"\nFull report saved to {output_path}")

if __name__ == "__main__":
    main()
