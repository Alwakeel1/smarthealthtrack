"""
SmartHealth-Track Package
-------------------------
A library for generating and accessing synthetic datasets for infectious disease monitoring.
"""

from smarthealthtrack.datasets import (
    load_patient_records,
    load_pharmaceutical_sales,
    load_wearable_health_data,
    load_wastewater_surveillance,
    generate_all_datasets
)

__version__ = '0.1.0'
