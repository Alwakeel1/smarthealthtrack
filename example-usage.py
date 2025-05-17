

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import from smarthealthtrack package
from smarthealthtrack import (
    load_patient_records, 
    load_pharmaceutical_sales,
    load_wearable_health_data,
    load_wastewater_surveillance,
    generate_all_datasets
)

def main():
    # Option 1: Generate all datasets
    print("Generating all datasets...")
    datasets = generate_all_datasets(save_to_csv=True)
    
    # Option 2: Load individual datasets
    print("\nLoading datasets individually...")
    patient_df = load_patient_records()
    pharma_df = load_pharmaceutical_sales()
    wearable_df = load_wearable_health_data()
    wastewater_df = load_wastewater_surveillance()
    
    # Print basic information about each dataset
    print("\nDataset Information:")
    print("-" * 50)
    
    print("Patient Records:")
    print(f"  Shape: {patient_df.shape}")
    print(f"  Columns: {', '.join(patient_df.columns)}")
    print(f"  Date Range: {patient_df['Timestamp'].min().date()} to {patient_df['Timestamp'].max().date()}")
    print(f"  Unique Diagnoses: {', '.join(patient_df['Diagnosis'].unique())}")
    
    print("\nPharmaceutical Sales:")
    print(f"  Shape: {pharma_df.shape}")
    print(f"  Columns: {', '.join(pharma_df.columns)}")
    print(f"  Date Range: {pharma_df['Date'].min().date()} to {pharma_df['Date'].max().date()}")
    print(f"  Unique Medications: {len(pharma_df['Medication'].unique())}")
    
    print("\nWearable Health Data:")
    print(f"  Shape: {wearable_df.shape}")
    print(f"  Columns: {', '.join(wearable_df.columns)}")
    print(f"  Date Range: {wearable_df['Timestamp'].min().date()} to {wearable_df['Timestamp'].max().date()}")
    print(f"  Unique Patients: {wearable_df['Patient_ID'].nunique()}")
    print(f"  Fever Detections: {wearable_df['Fever_Detected'].sum()} ({wearable_df['Fever_Detected'].mean()*100:.2f}%)")
    
    print("\nWastewater Surveillance:")
    print(f"  Shape: {wastewater_df.shape}")
    print(f"  Columns: {', '.join(wastewater_df.columns)}")
    print(f"  Date Range: {wastewater_df['Date'].min().date()} to {wastewater_df['Date'].max().date()}")
    print(f"  Pathogens Monitored: {', '.join(wastewater_df['Pathogen'].unique())}")
    print(f"  Sampling Locations: {', '.join(wastewater_df['Location'].unique())}")
    
    # Example analysis: Outbreak detection from pharmaceutical sales
    print("\nExample Analysis: Detecting outbreaks from pharmaceutical sales")
    tamiflu_sales = pharma_df[pharma_df['Medication'] == 'Tamiflu']
    tamiflu_ts = tamiflu_sales.set_index('Date')['Units_Sold']
    
    # Identify sales spikes (simple threshold-based anomaly detection)
    threshold = tamiflu_ts.mean() + 2 * tamiflu_ts.std()
    anomalies = tamiflu_ts[tamiflu_ts > threshold]
    
    print(f"Detected {len(anomalies)} potential outbreak days based on Tamiflu sales spikes")
    print(f"Anomaly dates: {', '.join([str(date.date()) for date in anomalies.index[:5]])}...")
    
    # Display success message
    print("\nSmartHealth-Track datasets successfully loaded and analyzed!")

if __name__ == "__main__":
    main()
