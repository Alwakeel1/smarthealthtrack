# SmartHealth-Track Datasets

This Python library provides synthetic datasets for research on AI-assisted real-time monitoring of infectious diseases in urban areas. The datasets are designed to match the requirements described in the paper "AI-Assisted Real-Time Monitoring of Infectious Diseases in Urban Areas" by Mohammed M. Alwakeel.
(Read the file named: "Data and Datasets Collection .pdf" included in this repository for more information)

## Installation

```bash
# Clone the repository
git clone https://github.com/alwakeel1/smarthealthtrack.git
cd smarthealthtrack

# Install the package
pip install -e .
```

## Available Datasets

The library provides four types of synthetic datasets:

1. **Patient Records** - Contains patient demographics, symptoms, diagnoses, and outcomes
2. **Pharmaceutical Sales** - Daily sales data for various medications
3. **Wearable Health Data** - Body temperature and heart rate measurements from wearable devices  
4. **Wastewater Surveillance** - Pathogen concentration data from wastewater monitoring

## Usage

### Loading Datasets

```python
from smarthealthtrack import (
    load_patient_records,
    load_pharmaceutical_sales, 
    load_wearable_health_data,
    load_wastewater_surveillance
)

# Load datasets
patient_df = load_patient_records()
pharma_df = load_pharmaceutical_sales()
wearable_df = load_wearable_health_data()
wastewater_df = load_wastewater_surveillance()
```

### Generating New Datasets

```python
from smarthealthtrack import generate_all_datasets

# Generate all datasets and save to CSV
datasets = generate_all_datasets(save_to_csv=True)

# Access individual datasets from the returned dictionary
patient_df = datasets['patient_records']
pharma_df = datasets['pharmaceutical_sales']
wearable_df = datasets['wearable_health_data']
wastewater_df = datasets['wastewater_surveillance']
```

## Dataset Details

### Patient Records Dataset

Columns:
- `Patient_ID`: Unique identifier for each patient
- `Age`: Age of the patient
- `Gender`: Gender of the patient (Male, Female, Other)
- `Latitude`, `Longitude`: Geospatial coordinates
- `Timestamp`: Date and time of record entry
- `Symptoms`: Reported symptoms (comma-separated)
- `Diagnosis`: Disease classification
- `Medication`: Prescribed medications
- `Outcome`: Patient outcome (Recovery, Hospitalization, Fatality)

### Pharmaceutical Sales Dataset

Columns:
- `Date`: Date of sale
- `Medication`: Name of medication
- `Units_Sold`: Number of units sold

### Wearable Health Data Dataset

Columns:
- `Patient_ID`: Unique identifier for each patient
- `Timestamp`: Date and time of measurement
- `Body_Temperature`: Body temperature in Celsius
- `Heart_Rate`: Heart rate in beats per minute
- `Fever_Detected`: Boolean flag indicating if fever was detected

### Wastewater Surveillance Dataset

Columns:
- `Date`: Date of sample collection
- `Location`: Name of wastewater treatment plant
- `Latitude`, `Longitude`: Geospatial coordinates of the plant
- `Pathogen`: Type of pathogen detected
- `Concentration_copies_per_mL`: Pathogen concentration in the sample
- `Population_Served`: Population served by the treatment plant
- `Outbreak_Period`: Boolean flag indicating if the date falls within an outbreak period

## Example

See `example_usage.py` for a complete example of how to load and use these datasets.
