"""
SmartHealth-Track Dataset Library
---------------------------------
A library for generating and accessing synthetic datasets for infectious disease monitoring
as described in the paper "AI-Assisted Real-Time Monitoring of Infectious Diseases in Urban Areas".
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy.stats import norm

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Package data directory
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, 'data')

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Helper functions
def add_outbreak_effect(base_series, outbreak_dates, outbreak_intensity=3, outbreak_duration=14):
    """
    Add outbreak effects to a time series at specified dates
    """
    series = base_series.copy()
    for outbreak_date in outbreak_dates:
        # Convert string date to datetime if needed
        if isinstance(outbreak_date, str):
            outbreak_date = pd.to_datetime(outbreak_date)
        
        # Create gaussian-shaped outbreak effect
        for i in range(-outbreak_duration//2, outbreak_duration//2):
            day = outbreak_date + timedelta(days=i)
            if day in series.index:
                # Apply Gaussian shape with peak at outbreak date
                effect = outbreak_intensity * np.exp(-(i**2) / (outbreak_duration/2))
                series[day] *= (1 + effect/10)
    
    return series

def generate_location_data(num_points, center_lat=24.7136, center_lon=46.6753, radius=0.1):
    """Generate geospatial data centered around a specific point with random spread"""
    lats = np.random.normal(center_lat, radius, num_points)
    lons = np.random.normal(center_lon, radius, num_points)
    return lats, lons

# Dataset generation functions
def generate_patient_records(num_patients=1000, start_date='2023-01-01', end_date='2023-12-31'):
    """
    Generate synthetic patient records dataset
    """
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Define outbreak dates
    outbreak_dates = ['2023-01-15', '2023-04-10', '2023-07-20', '2023-10-05']
    outbreak_dates = [pd.to_datetime(date) for date in outbreak_dates]
    
    # Generate patient IDs
    patient_ids = range(1, num_patients + 1)
    
    # Generate ages with a realistic distribution
    ages = np.random.lognormal(mean=3.5, sigma=0.5, size=num_patients).astype(int)
    # Clip ages to a reasonable range
    ages = np.clip(ages, 1, 100)
    
    # Generate gender (0: Male, 1: Female, 2: Other)
    genders = np.random.choice(['Male', 'Female', 'Other'], num_patients, p=[0.48, 0.48, 0.04])
    
    # Generate locations (latitude, longitude)
    lats, lons = generate_location_data(num_patients)
    
    # Generate timestamps (admission dates)
    # More patients during outbreak periods
    timestamps = []
    for _ in range(num_patients):
        # 70% chance to be near an outbreak, 30% chance to be random
        if np.random.random() < 0.7:
            # Pick a random outbreak
            outbreak_date = np.random.choice(outbreak_dates)
            # Add some days before or after the outbreak (-10 to +20 days)
            days_offset = np.random.randint(-10, 21)
            timestamp = outbreak_date + timedelta(days=days_offset)
            if timestamp < start_date or timestamp > end_date:
                timestamp = np.random.choice(date_range)
        else:
            timestamp = np.random.choice(date_range)
        timestamps.append(timestamp)
    
    # Generate symptoms
    common_symptoms = ['Fever', 'Cough', 'Fatigue', 'Shortness of Breath', 'Headache', 
                      'Sore Throat', 'Nausea', 'Body Aches', 'Diarrhea', 'Vomiting']
    
    symptoms = []
    for timestamp in timestamps:
        # Patients during outbreaks have more symptoms
        is_near_outbreak = any(abs((timestamp - outbreak).days) < 15 for outbreak in outbreak_dates)
        
        if is_near_outbreak:
            num_symptoms = np.random.randint(2, 6)  # 2-5 symptoms during outbreaks
        else:
            num_symptoms = np.random.randint(1, 4)  # 1-3 symptoms normally
            
        patient_symptoms = ', '.join(np.random.choice(common_symptoms, min(num_symptoms, len(common_symptoms)), replace=False))
        symptoms.append(patient_symptoms)
    
    # Generate diagnoses
    diagnoses = []
    for timestamp in timestamps:
        is_near_outbreak = any(abs((timestamp - outbreak).days) < 15 for outbreak in outbreak_dates)
        
        if is_near_outbreak:
            disease = np.random.choice(['Influenza', 'COVID-19', 'Respiratory Infection', 'Pneumonia'], 
                                     p=[0.3, 0.4, 0.2, 0.1])
        else:
            disease = np.random.choice(['Influenza', 'Common Cold', 'Allergic Reaction', 'Stomach Virus', 'Bronchitis'],
                                      p=[0.1, 0.3, 0.25, 0.25, 0.1])
        diagnoses.append(disease)
    
    # Generate medications
    medications = []
    for diagnosis in diagnoses:
        if diagnosis == 'Influenza':
            med = np.random.choice(['Tamiflu', 'Xofluza', 'Relenza', 'Paracetamol + Rest'])
        elif diagnosis == 'COVID-19':
            med = np.random.choice(['Paxlovid', 'Molnupiravir', 'Dexamethasone', 'Remdesivir'])
        elif diagnosis == 'Common Cold':
            med = np.random.choice(['Paracetamol', 'Ibuprofen', 'Sudafed', 'Nyquil'])
        elif diagnosis == 'Allergic Reaction':
            med = np.random.choice(['Cetirizine', 'Loratadine', 'Benadryl', 'Prednisone'])
        elif diagnosis == 'Stomach Virus':
            med = np.random.choice(['Ondansetron', 'Loperamide', 'Pepto-Bismol', 'Electrolyte solution'])
        elif diagnosis == 'Bronchitis':
            med = np.random.choice(['Azithromycin', 'Amoxicillin', 'Dextromethorphan', 'Benzonatate'])
        elif diagnosis == 'Respiratory Infection':
            med = np.random.choice(['Azithromycin', 'Levofloxacin', 'Amoxicillin', 'Doxycycline'])
        elif diagnosis == 'Pneumonia':
            med = np.random.choice(['Azithromycin + Ceftriaxone', 'Levofloxacin', 'Moxifloxacin', 'Piperacillin-Tazobactam'])
        else:
            med = 'Symptomatic treatment'
        medications.append(med)
    
    # Generate outcomes
    outcomes = []
    for diagnosis, timestamp in zip(diagnoses, timestamps):
        is_near_outbreak = any(abs((timestamp - outbreak).days) < 15 for outbreak in outbreak_dates)
        severity = np.random.random()
        
        if diagnosis in ['COVID-19', 'Pneumonia'] and severity > 0.7:
            outcome = 'Hospitalization'
        elif diagnosis in ['Influenza', 'Respiratory Infection'] and severity > 0.85:
            outcome = 'Hospitalization'
        elif is_near_outbreak and severity > 0.9:
            outcome = 'Hospitalization'
        elif severity > 0.98:  # Very rare fatality
            outcome = 'Fatality'
        else:
            outcome = 'Recovery'
        
        outcomes.append(outcome)
    
    # Create the dataframe
    patient_df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Age': ages,
        'Gender': genders,
        'Latitude': lats,
        'Longitude': lons,
        'Timestamp': timestamps,
        'Symptoms': symptoms,
        'Diagnosis': diagnoses,
        'Medication': medications,
        'Outcome': outcomes
    })
    
    return patient_df

def generate_pharmaceutical_sales(start_date='2023-01-01', end_date='2023-12-31'):
    """
    Generate synthetic pharmaceutical sales data
    """
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Define common medications
    medications = [
        'Tamiflu', 'Xofluza', 'Relenza',  # Influenza
        'Paxlovid', 'Molnupiravir',  # COVID-19
        'Paracetamol', 'Ibuprofen', 'Sudafed',  # Common Cold
        'Cetirizine', 'Loratadine', 'Benadryl',  # Allergies
        'Ondansetron', 'Loperamide', 'Pepto-Bismol',  # Stomach issues
        'Azithromycin', 'Amoxicillin', 'Doxycycline'  # Antibiotics
    ]
    
    # Define outbreak dates
    outbreak_dates = ['2023-01-15', '2023-04-10', '2023-07-20', '2023-10-05']
    
    # Generate daily baseline sales for each medication
    sales_data = {}
    
    for medication in medications:
        # Base sales with weekly patterns (higher on weekdays, lower on weekends)
        base_sales = np.array([100 + 20 * np.sin(2 * np.pi * i / 7) for i in range(len(date_range))])
        
        # Add seasonal trends for certain medications
        if medication in ['Tamiflu', 'Xofluza', 'Relenza']:
            # Higher in winter (assuming Northern Hemisphere where Jan is winter)
            seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * (np.arange(len(date_range)) - 15) / 365)
            base_sales *= seasonal_factor
        
        if medication in ['Cetirizine', 'Loratadine', 'Benadryl']:
            # Higher in spring/summer
            seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * (np.arange(len(date_range)) - 100) / 365)
            base_sales *= seasonal_factor
        
        # Add random noise
        noise = np.random.normal(0, 0.05 * base_sales.mean(), len(date_range))
        sales = base_sales + noise
        
        # Add outbreak effects for relevant medications
        if medication in ['Tamiflu', 'Paracetamol', 'Ibuprofen', 'Azithromycin']:
            # Stronger effect on these medications during outbreaks
            sales_series = pd.Series(sales, index=date_range)
            sales_series = add_outbreak_effect(sales_series, outbreak_dates, 
                                              outbreak_intensity=5 if medication == 'Tamiflu' else 3)
            sales = sales_series.values
        
        # Ensure sales are positive
        sales = np.maximum(sales, 0)
        
        # Add to dictionary
        sales_data[medication] = sales.astype(int)
    
    # Create dataframe
    sales_df = pd.DataFrame(sales_data, index=date_range)
    sales_df.index.name = 'Date'
    
    # Melt dataframe for easier analysis
    sales_long_df = sales_df.reset_index().melt(id_vars='Date', var_name='Medication', value_name='Units_Sold')
    
    return sales_long_df

def generate_wearable_health_data(patient_df, start_date='2023-01-01', end_date='2023-12-31'):
    """
    Generate synthetic wearable health data for patients
    """
    # Get a subset of patients who would have wearables (assume 30% of patients)
    num_wearable_users = int(len(patient_df) * 0.3)
    wearable_users = patient_df.sample(n=num_wearable_users)
    
    # Parameters for normal body temperature and heart rate
    normal_temp_mean, normal_temp_std = 36.8, 0.3  # in Celsius
    normal_hr_mean, normal_hr_std = 72, 8  # in BPM
    
    # Define the fever threshold (in Celsius)
    fever_threshold = 38.0
    
    # Data collection frequency (readings per day)
    readings_per_day = 6  # every 4 hours
    
    # Generate data
    wearable_records = []
    
    for _, patient in wearable_users.iterrows():
        # Get patient info
        patient_id = patient['Patient_ID']
        diagnosis = patient['Diagnosis']
        admission_date = patient['Timestamp']
        
        # Determine the monitoring period (start 5 days before admission, continue 10 days after)
        monitor_start = max(pd.to_datetime(start_date), admission_date - timedelta(days=5))
        monitor_end = min(pd.to_datetime(end_date), admission_date + timedelta(days=10))
        
        # Generate timestamps for the monitoring period
        monitoring_hours = int((monitor_end - monitor_start).total_seconds() / 3600)
        reading_timestamps = [monitor_start + timedelta(hours=h) for h in range(0, monitoring_hours, 24//readings_per_day)]
        
        for timestamp in reading_timestamps:
            # Calculate days from admission
            days_from_admission = (timestamp - admission_date).total_seconds() / (24 * 3600)
            
            # Determine if patient is showing symptoms based on proximity to admission date
            # More likely to have symptoms close to and after admission
            is_symptomatic = False
            symptom_probability = 0.0
            
            if diagnosis in ['Influenza', 'COVID-19', 'Pneumonia', 'Respiratory Infection']:
                # More serious conditions have more pronounced symptoms
                if -2 <= days_from_admission < 0:  # 2 days before admission
                    symptom_probability = 0.3 + 0.3 * (2 + days_from_admission)  # Increasing from 0.3 to 0.9
                elif 0 <= days_from_admission < 5:  # First 5 days after admission
                    symptom_probability = 0.9 - 0.15 * days_from_admission  # Decreasing from 0.9 to 0.15
                elif 5 <= days_from_admission <= 10:  # Days 5-10 after admission
                    symptom_probability = 0.15 - 0.015 * (days_from_admission - 5)  # Decreasing from 0.15 to 0
            else:
                # Less serious conditions have milder symptoms
                if -1 <= days_from_admission < 0:  # 1 day before admission
                    symptom_probability = 0.2 + 0.5 * (1 + days_from_admission)  # Increasing from 0.2 to 0.7
                elif 0 <= days_from_admission < 3:  # First 3 days after admission
                    symptom_probability = 0.7 - 0.2 * days_from_admission  # Decreasing from 0.7 to 0.1
                elif 3 <= days_from_admission <= 5:  # Days 3-5 after admission
                    symptom_probability = 0.1 - 0.05 * (days_from_admission - 3)  # Decreasing from 0.1 to 0
            
            is_symptomatic = np.random.random() < symptom_probability
            
            # Generate temperature based on whether patient is symptomatic
            if is_symptomatic:
                # Higher temperature for symptomatic patients
                temp_mean = normal_temp_mean + np.random.uniform(1.0, 2.0)  # 1-2°C higher
                temp_std = normal_temp_std * 1.2  # More variable
                
                # Generate heart rate (higher when fevered)
                hr_mean = normal_hr_mean + np.random.uniform(15, 30)  # 15-30 BPM higher
                hr_std = normal_hr_std * 1.5  # More variable
            else:
                # Normal temperature range
                temp_mean = normal_temp_mean
                temp_std = normal_temp_std
                
                # Normal heart rate
                hr_mean = normal_hr_mean
                hr_std = normal_hr_std
            
            # Add daily and individual variation
            time_of_day = timestamp.hour
            if 0 <= time_of_day < 6:  # Early morning: lower
                temp_offset = -0.3
                hr_offset = -5
            elif 12 <= time_of_day < 18:  # Afternoon: higher
                temp_offset = 0.2
                hr_offset = 5
            else:  # Morning/Evening: normal
                temp_offset = 0
                hr_offset = 0
            
            # Generate temperature with added individual variation
            temperature = np.random.normal(temp_mean + temp_offset, temp_std)
            
            # Generate heart rate with added individual variation
            heart_rate = np.random.normal(hr_mean + hr_offset, hr_std)
            
            # Clip values to physiologically plausible ranges
            temperature = np.clip(temperature, 35.0, 41.0)
            heart_rate = np.clip(heart_rate, 40, 200)
            
            # Add the record
            wearable_records.append({
                'Patient_ID': patient_id,
                'Timestamp': timestamp,
                'Body_Temperature': round(temperature, 1),  # Round to 1 decimal place
                'Heart_Rate': int(heart_rate),
                'Fever_Detected': temperature >= fever_threshold
            })
    
    # Create dataframe
    wearable_df = pd.DataFrame(wearable_records)
    
    return wearable_df

def generate_wastewater_data(start_date='2023-01-01', end_date='2023-12-31'):
    """
    Generate synthetic wastewater surveillance data
    """
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define sampling locations (wastewater treatment plants or collection points)
    locations = [
        {'name': 'Central WWTP', 'lat': 24.71, 'lon': 46.67, 'population': 500000},
        {'name': 'North WWTP', 'lat': 24.85, 'lon': 46.71, 'population': 300000},
        {'name': 'South WWTP', 'lat': 24.62, 'lon': 46.73, 'population': 250000},
        {'name': 'East WWTP', 'lat': 24.75, 'lon': 46.85, 'population': 200000},
        {'name': 'West WWTP', 'lat': 24.73, 'lon': 46.60, 'population': 180000}
    ]
    
    # Define pathogens to monitor
    pathogens = ['SARS-CoV-2', 'Influenza A', 'Influenza B', 'Norovirus', 'Rotavirus']
    
    # Define outbreak dates
    outbreak_dates = ['2023-01-15', '2023-04-10', '2023-07-20', '2023-10-05']
    
    # Generate data
    wastewater_records = []
    
    for location in locations:
        for date in date_range:
            for pathogen in pathogens:
                # Base concentration (copies per mL)
                if pathogen == 'SARS-CoV-2':
                    base_conc = 1000
                elif pathogen in ['Influenza A', 'Influenza B']:
                    # Seasonal pattern for influenza
                    day_of_year = date.dayofyear
                    # Peak in winter (day 15 = Jan 15)
                    seasonal_factor = 1 + 3 * np.exp(-0.5 * ((day_of_year - 15) % 365)**2 / 5000)
                    base_conc = 800 * seasonal_factor
                elif pathogen == 'Norovirus':
                    # Winter peak for norovirus
                    day_of_year = date.dayofyear
                    seasonal_factor = 1 + 2 * np.exp(-0.5 * ((day_of_year - 30) % 365)**2 / 4000)
                    base_conc = 600 * seasonal_factor
                else:  # Rotavirus
                    # More stable year-round
                    base_conc = 500
                
                # Add population size effect (larger populations have higher concentrations)
                population_factor = location['population'] / 200000
                base_conc *= population_factor
                
                # Add random variation
                variation = np.random.normal(0, 0.1 * base_conc)
                concentration = base_conc + variation
                
                # Add outbreak effects
                is_outbreak = False
                for outbreak_date in outbreak_dates:
                    outbreak_date = pd.to_datetime(outbreak_date)
                    days_diff = abs((date - outbreak_date).days)
                    
                    if days_diff <= 14:  # Within 2 weeks of outbreak
                        if pathogen in ['SARS-CoV-2', 'Influenza A'] and days_diff <= 7:
                            # Strong effect for these pathogens at early outbreak
                            outbreak_multiplier = 5.0 * np.exp(-0.2 * days_diff)
                            concentration *= outbreak_multiplier
                            is_outbreak = True
                        elif pathogen in ['Influenza B', 'Norovirus'] and days_diff <= 10:
                            # Moderate effect for these pathogens
                            outbreak_multiplier = 3.0 * np.exp(-0.15 * days_diff)
                            concentration *= outbreak_multiplier
                            is_outbreak = True
                
                # Ensure concentration is positive
                concentration = max(concentration, 0)
                
                # Add the record
                wastewater_records.append({
                    'Date': date,
                    'Location': location['name'],
                    'Latitude': location['lat'],
                    'Longitude': location['lon'],
                    'Pathogen': pathogen,
                    'Concentration_copies_per_mL': round(concentration, 2),
                    'Population_Served': location['population'],
                    'Outbreak_Period': is_outbreak
                })
    
    # Create dataframe
    wastewater_df = pd.DataFrame(wastewater_records)
    
    return wastewater_df

def generate_all_datasets(save_to_csv=True):
    """
    Generate all synthetic datasets for the SmartHealth-Track system
    """
    print("Generating patient records...")
    patient_df = generate_patient_records()
    
    print("Generating pharmaceutical sales data...")
    pharma_df = generate_pharmaceutical_sales()
    
    print("Generating wearable health data...")
    wearable_df = generate_wearable_health_data(patient_df)
    
    print("Generating wastewater surveillance data...")
    wastewater_df = generate_wastewater_data()
    
    if save_to_csv:
        print("Saving datasets to CSV files...")
        patient_df.to_csv(os.path.join(DATA_DIR, 'patient_records.csv'), index=False)
        pharma_df.to_csv(os.path.join(DATA_DIR, 'pharmaceutical_sales.csv'), index=False)
        wearable_df.to_csv(os.path.join(DATA_DIR, 'wearable_health_data.csv'), index=False)
        wastewater_df.to_csv(os.path.join(DATA_DIR, 'wastewater_surveillance.csv'), index=False)
    
    print("Dataset generation complete!")
    
    return {
        'patient_records': patient_df,
        'pharmaceutical_sales': pharma_df,
        'wearable_health_data': wearable_df,
        'wastewater_surveillance': wastewater_df
    }

# API functions to load datasets
def load_patient_records():
    """Load the patient records dataset"""
    csv_path = os.path.join(DATA_DIR, 'patient_records.csv')
    
    # Generate if not exists
    if not os.path.exists(csv_path):
        print("Patient records dataset not found. Generating dataset...")
        datasets = generate_all_datasets()
        return datasets['patient_records']
    
    # Load from CSV
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'])
    return df

def load_pharmaceutical_sales():
    """Load the pharmaceutical sales dataset"""
    csv_path = os.path.join(DATA_DIR, 'pharmaceutical_sales.csv')
    
    # Generate if not exists
    if not os.path.exists(csv_path):
        print("Pharmaceutical sales dataset not found. Generating dataset...")
        datasets = generate_all_datasets()
        return datasets['pharmaceutical_sales']
    
    # Load from CSV
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    return df

def load_wearable_health_data():
    """Load the wearable health data dataset"""
    csv_path = os.path.join(DATA_DIR, 'wearable_health_data.csv')
    
    # Generate if not exists
    if not os.path.exists(csv_path):
        print("Wearable health data dataset not found. Generating dataset...")
        datasets = generate_all_datasets()
        return datasets['wearable_health_data']
    
    # Load from CSV
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'])
    return df

def load_wastewater_surveillance():
    """Load the wastewater surveillance dataset"""
    csv_path = os.path.join(DATA_DIR, 'wastewater_surveillance.csv')
    
    # Generate if not exists
    if not os.path.exists(csv_path):
        print("Wastewater surveillance dataset not found. Generating dataset...")
        datasets = generate_all_datasets()
        return datasets['wastewater_surveillance']
    
    # Load from CSV
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    return df

# Generate datasets if this file is run directly
if __name__ == "__main__":
    generate_all_datasets(save_to_csv=True)
