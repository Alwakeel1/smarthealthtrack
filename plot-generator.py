"""
Generate all plots from the paper "AI-Assisted Real-Time Monitoring of Infectious Diseases in Urban Areas".
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Create directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Import dataset functions
from smarthealthtrack import (
    load_patient_records,
    load_pharmaceutical_sales,
    load_wearable_health_data,
    load_wastewater_surveillance,
    generate_all_datasets
)

# Load or generate datasets
try:
    patient_df = load_patient_records()
    pharma_df = load_pharmaceutical_sales()
    wearable_df = load_wearable_health_data()
    wastewater_df = load_wastewater_surveillance()
except:
    print("Generating datasets first...")
    datasets = generate_all_datasets(save_to_csv=True)
    patient_df = datasets['patient_records']
    pharma_df = datasets['pharmaceutical_sales']
    wearable_df = datasets['wearable_health_data']
    wastewater_df = datasets['wastewater_surveillance']

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Helper functions
def save_plot(fig, filename):
    """Save plot to the plots directory with high resolution"""
    fig.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)

# --------------------------------
# Figure 2: Performance Evaluation
# --------------------------------
def plot_performance_evaluation():
    # Model performance data from the paper
    models = ['SmartHealth-Track\n(Logistic Regression)', 
              'SmartHealth-Track\n(Random Forest)', 
              'SmartHealth-Track\n(LSTM)', 
              'Baseline Model\n(Statistical Forecasting)']
    
    accuracy = [89.4, 92.3, 94.8, 78.2]
    precision = [0.87, 0.91, 0.93, 0.76]
    recall = [0.88, 0.90, 0.94, 0.74]
    f1_score = [0.875, 0.905, 0.935, 0.75]
    rmse = [2.15, 1.84, 1.52, 3.65]
    
    # Normalize RMSE for visualization (lower is better)
    max_rmse = max(rmse)
    normalized_rmse = [1 - (r / max_rmse * 0.8) for r in rmse]  # Scale to keep bars visible
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.15
    index = np.arange(len(models))
    
    # Plot bars
    ax1.bar(index - 2*bar_width, accuracy, bar_width, label='Accuracy (%)', color='#3274A1')
    ax1.bar(index - bar_width, [p*100 for p in precision], bar_width, label='Precision (%)', color='#E1812C')
    ax1.bar(index, [r*100 for r in recall], bar_width, label='Recall (%)', color='#3A923A')
    ax1.bar(index + bar_width, [f*100 for f in f1_score], bar_width, label='F1-Score (%)', color='#C03D3E')
    
    # Create a secondary y-axis for RMSE (lower is better)
    ax2 = ax1.twinx()
    ax2.bar(index + 2*bar_width, rmse, bar_width, label='RMSE', color='#9372B2')
    ax2.set_ylabel('RMSE (lower is better)', fontsize=12)
    ax2.set_ylim(0, 4)
    
    # Customize primary axis
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_ylim(70, 100)
    ax1.set_xticks(index)
    ax1.set_xticklabels(models, rotation=0, ha='center')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    # Add title
    plt.title('Performance Evaluation of the SmartHealth-Track Model', fontsize=16, pad=20)
    plt.tight_layout()
    
    save_plot(fig, 'figure2_performance_evaluation.png')
    return fig

# --------------------------------
# Figure 3: Confusion Matrix
# --------------------------------
def plot_confusion_matrix():
    # Confusion matrix data from the paper
    cm = np.array([
        [115, 12],  # Anomaly Present: 115 TP, 12 FN
        [9, 864]    # No Anomaly: 9 FP, 864 TN
    ])
    
    # Create custom colormap (blue for TP/TN, light colors for errors)
    colors = ['#4878D0', '#EE854A', '#EE854A', '#4878D0']
    cm_colors = LinearSegmentedColormap.from_list('custom_cmap', ['#D6EAF8', '#2874A6'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=cm_colors, 
                xticklabels=['Anomaly Detected', 'No Anomaly'],
                yticklabels=['Anomaly Present', 'No Anomaly'],
                linewidths=0.5, linecolor='gray', cbar=False)
    
    # Calculate and display metrics
    total = np.sum(cm)
    accuracy = (cm[0,0] + cm[1,1]) / total
    precision = cm[0,0] / (cm[0,0] + cm[1,0])
    recall = cm[0,0] / (cm[0,0] + cm[0,1])
    f1 = 2 * precision * recall / (precision + recall)
    
    info_text = (f"Accuracy: {accuracy:.1%}\n"
                 f"Precision: {precision:.1%}\n"
                 f"Recall: {recall:.1%}\n"
                 f"F1 Score: {f1:.1%}")
    
    # Add info box
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax.text(1.05, 0.5, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=props)
    
    plt.title('Confusion Matrix for Anomaly Detection in Pharmaceutical Demand Patterns', fontsize=14, pad=20)
    plt.tight_layout()
    
    save_plot(fig, 'figure3_confusion_matrix.png')
    return fig

# --------------------------------
# Figure 4: Actual vs Predicted Outbreaks
# --------------------------------
def plot_outbreak_comparison():
    # Outbreak data from the paper
    time_periods = ['Jan-Mar 2023', 'Apr-Jun 2023', 'Jul-Sep 2023', 'Oct-Dec 2023']
    actual_outbreaks = [4, 6, 9, 7]
    predicted_outbreaks = [3, 5, 8, 6]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(time_periods))
    
    # Plot bars
    ax.bar(index - bar_width/2, actual_outbreaks, bar_width, label='Actual Outbreaks', color='#3274A1')
    ax.bar(index + bar_width/2, predicted_outbreaks, bar_width, label='Predicted Outbreaks', color='#E1812C')
    
    # Add count numbers on top of bars
    for i, v in enumerate(actual_outbreaks):
        ax.text(i - bar_width/2, v + 0.1, str(v), ha='center', fontsize=10)
    
    for i, v in enumerate(predicted_outbreaks):
        ax.text(i + bar_width/2, v + 0.1, str(v), ha='center', fontsize=10)
    
    # Customize axis
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Number of Outbreaks', fontsize=12)
    ax.set_xticks(index)
    ax.set_xticklabels(time_periods)
    
    # Add legend and title
    ax.legend()
    plt.title('Comparison of Actual and Predicted Outbreaks over Different Time Periods', fontsize=14, pad=20)
    plt.tight_layout()
    
    save_plot(fig, 'figure4_outbreak_comparison.png')
    return fig

# --------------------------------
# Figure 5: Wearable Device Monitoring Results
# --------------------------------
def plot_wearable_monitoring():
    # Extract data from wearable dataset
    fever_by_day = wearable_df.groupby(wearable_df['Timestamp'].dt.date)['Fever_Detected'].mean() * 100
    date_range = fever_by_day.index
    
    # Calculate rolling average (7-day window)
    fever_rolling_avg = fever_by_day.rolling(window=7, min_periods=1).mean()
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot fever detection percentage
    color = '#3274A1'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Percentage of Users with Fever (%)', color=color, fontsize=12)
    ax1.plot(date_range, fever_by_day, color=color, alpha=0.3, label='Daily')
    ax1.plot(date_range, fever_rolling_avg, color=color, linewidth=2.5, label='7-day average')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add secondary y-axis for heart rate
    ax2 = ax1.twinx()
    color = '#E1812C'
    
    # Calculate average heart rate by day
    avg_hr_by_day = wearable_df.groupby(wearable_df['Timestamp'].dt.date)['Heart_Rate'].mean()
    avg_hr_rolling = avg_hr_by_day.rolling(window=7, min_periods=1).mean()
    
    ax2.set_ylabel('Average Heart Rate (BPM)', color=color, fontsize=12)
    ax2.plot(date_range, avg_hr_by_day, color=color, alpha=0.3, linestyle='--')
    ax2.plot(date_range, avg_hr_rolling, color=color, linewidth=2.5, linestyle='--', label='Avg Heart Rate (7-day avg)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add outbreak markers
    outbreak_dates = [pd.to_datetime('2023-01-15'), pd.to_datetime('2023-04-10'), 
                      pd.to_datetime('2023-07-20'), pd.to_datetime('2023-10-05')]
    
    for outbreak_date in outbreak_dates:
        plt.axvline(x=outbreak_date, color='red', linestyle='--', alpha=0.7)
        ax1.text(outbreak_date, ax1.get_ylim()[1] * 0.95, 'Outbreak', 
                rotation=90, verticalalignment='top', fontsize=10, color='red')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add accuracy metrics as text box
    metrics_text = (
        "Fever Detection Accuracy: 93.5%\n"
        "Anomalous Heart Rate Detection: 91.8%\n"
        "False Positive Rate: 5.4%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    plt.figtext(0.15, 0.15, metrics_text, fontsize=10, 
                verticalalignment='bottom', bbox=props)
    
    plt.title('Wearable Device Monitoring Results', fontsize=14, pad=20)
    plt.tight_layout()
    
    save_plot(fig, 'figure5_wearable_monitoring.png')
    return fig

# --------------------------------
# Figure 6: AI-Based Early Disease Outbreak Detection
# --------------------------------
def plot_early_outbreak_detection():
    # Create synthetic outbreak detection data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31')
    
    # Base outbreak probability
    base_prob = np.zeros(len(dates))
    
    # Add outbreak effects
    outbreak_dates = [pd.to_datetime('2023-01-15'), pd.to_datetime('2023-04-10'), 
                     pd.to_datetime('2023-07-20'), pd.to_datetime('2023-10-05')]
    
    for outbreak_date in outbreak_dates:
        idx = (dates == outbreak_date).argmax()
        # Create Gaussian curve around outbreak date
        for i in range(max(0, idx-30), min(len(dates), idx+30)):
            distance = abs((i - idx) / 10)
            base_prob[i] += 0.8 * np.exp(-distance**2)
    
    # Add noise
    noise = np.random.normal(0, 0.05, len(dates))
    detection_prob = np.clip(base_prob + noise, 0, 1)
    
    # Create alert dates (when probability exceeds threshold)
    threshold = 0.4
    alerts = detection_prob > threshold
    
    # Simulate healthcare metrics affected by outbreaks
    hospital_admissions = 50 + 150 * base_prob + np.random.normal(0, 10, len(dates))
    hospital_admissions = np.clip(hospital_admissions, 0, None).astype(int)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot outbreak probability
    ax1.plot(dates, detection_prob, 'b-', label='Outbreak Probability')
    ax1.axhline(y=threshold, color='r', linestyle='--', label='Alert Threshold')
    
    # Highlight alert periods
    alert_regions = np.where(alerts)[0]
    for i in alert_regions:
        ax1.axvspan(dates[i], dates[i] + pd.Timedelta(days=1), alpha=0.3, color='red')
    
    # Plot hospital admissions on secondary axis
    ax3 = ax1.twinx()
    ax3.plot(dates, hospital_admissions, 'g-', alpha=0.7, label='Hospital Admissions')
    ax3.set_ylabel('Daily Hospital Admissions', color='g')
    
    # Add vertical lines for actual outbreak dates
    for outbreak_date in outbreak_dates:
        ax1.axvline(x=outbreak_date, color='black', linestyle='-', alpha=0.7)
        ax1.text(outbreak_date, 0.95, 'Actual\nOutbreak', 
                rotation=90, verticalalignment='top', fontsize=10)
    
    # Customize first plot
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Outbreak Probability')
    ax1.set_title('AI-Based Early Disease Outbreak Detection', fontsize=14)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines3, labels1 + labels3, loc='upper right')
    
    # Second plot for detection metrics
    detection_metrics = pd.DataFrame({
        'Date': dates,
        'Social Media Signals': np.clip(detection_prob * 0.9 + np.random.normal(0, 0.1, len(dates)), 0, 1),
        'Pharmacy Sales Anomalies': np.clip(detection_prob * 0.85 + np.random.normal(0, 0.15, len(dates)), 0, 1),
        'Symptom Search Trends': np.clip(detection_prob * 0.8 + np.random.normal(0, 0.12, len(dates)), 0, 1)
    })
    
    # Plot detection metrics
    ax2.plot(detection_metrics['Date'], detection_metrics['Social Media Signals'], label='Social Media Signals')
    ax2.plot(detection_metrics['Date'], detection_metrics['Pharmacy Sales Anomalies'], label='Pharmacy Sales Anomalies')
    ax2.plot(detection_metrics['Date'], detection_metrics['Symptom Search Trends'], label='Symptom Search Trends')
    
    # Customize second plot
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Signal Strength')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    
    # Add metrics as text
    metrics_text = (
        "Detection Accuracy: 92.4%\n"
        "False Positive Rate: 7.2%\n"
        "Processing Time: 120ms"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax1.text(dates[30], 0.15, metrics_text, fontsize=10, 
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    save_plot(fig, 'figure6_early_outbreak_detection.png')
    return fig

# --------------------------------
# Figure 7: AI-Based Contact Tracing
# --------------------------------
def plot_contact_tracing():
    # Generate synthetic contact tracing data
    np.random.seed(43)
    dates = pd.date_range(start='2023-01-01', end='2023-03-31')
    
    # Base data
    data = pd.DataFrame({
        'Date': dates,
        'Close Contacts Detected': [
            int(100 + 30 * np.sin(i/10) + np.random.normal(0, 10)) 
            for i in range(len(dates))
        ],
        'Exposure Alerts Sent': [
            int(80 + 25 * np.sin(i/10) + np.random.normal(0, 8)) 
            for i in range(len(dates))
        ],
        'Quarantine Compliance (%)': [
            min(100, max(60, 85 + 10 * np.sin(i/15) + np.random.normal(0, 5))) 
            for i in range(len(dates))
        ]
    })
    
    # Spike during outbreaks
    outbreak_dates = [pd.to_datetime('2023-01-15'), pd.to_datetime('2023-02-20')]
    
    for outbreak_date in outbreak_dates:
        idx = (data['Date'] == outbreak_date).argmax()
        # Create gaussian peak around outbreak date
        for i in range(max(0, idx-10), min(len(data), idx+10)):
            distance = abs((i - idx) / 5)
            multiplier = 2 * np.exp(-distance**2)
            data.loc[i, 'Close Contacts Detected'] = int(data.loc[i, 'Close Contacts Detected'] * multiplier)
            data.loc[i, 'Exposure Alerts Sent'] = int(data.loc[i, 'Exposure Alerts Sent'] * multiplier)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot contacts and alerts
    ax1.plot(data['Date'], data['Close Contacts Detected'], 'b-', label='Close Contacts Detected')
    ax1.plot(data['Date'], data['Exposure Alerts Sent'], 'r-', label='Exposure Alerts Sent')
    
    # Mark outbreak dates
    for outbreak_date in outbreak_dates:
        ax1.axvline(x=outbreak_date, color='black', linestyle='--', alpha=0.7)
        ax1.text(outbreak_date, ax1.get_ylim()[1] * 0.95, 'Outbreak', 
                rotation=90, verticalalignment='top', fontsize=10)
    
    # Customize first plot
    ax1.set_ylabel('Count per Day')
    ax1.legend(loc='upper right')
    ax1.set_title('AI-Based Contact Tracing and Exposure Alerts', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot compliance rate
    ax2.plot(data['Date'], data['Quarantine Compliance (%)'], 'g-', label='Quarantine Compliance Rate')
    ax2.set_ylabel('Compliance Rate (%)')
    ax2.set_ylim(50, 100)
    ax2.legend(loc='lower right')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    # Calculate alert accuracy
    contacts = data['Close Contacts Detected'].sum()
    alerts = data['Exposure Alerts Sent'].sum()
    accuracy = alerts / contacts * 100
    
    # Add metrics as text
    metrics_text = (
        f"Contact Detection Accuracy: 88.5%\n"
        f"Alert Precision: 91.2%\n"
        f"False Positive Rate: 8.6%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax1.text(data['Date'][10], 250, metrics_text, fontsize=10, 
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    save_plot(fig, 'figure7_contact_tracing.png')
    return fig

# --------------------------------
# Figure 8: AI-Driven Airborne Pathogen Detection
# --------------------------------
def plot_airborne_pathogen_detection():
    # Create synthetic airborne pathogen data
    np.random.seed(44)
    
    # Create time range for 24 hours monitoring
    times = pd.date_range(start='2023-02-01 00:00', end='2023-02-01 23:59', freq='15min')
    hours = [t.hour + t.minute/60 for t in times]
    
    # Generate pathogen concentration with a pattern of higher values during busy hours
    base_concentration = np.zeros(len(times))
    
    # Morning peak (7-9 AM)
    morning_mask = (hours >= 7) & (hours <= 9)
    base_concentration[morning_mask] = 5 + 10 * np.sin((hours[morning_mask] - 7) * np.pi / 2)
    
    # Lunch peak (12-2 PM)
    lunch_mask = (hours >= 12) & (hours <= 14)
    base_concentration[lunch_mask] = 6 + 8 * np.sin((hours[lunch_mask] - 12) * np.pi / 2)
    
    # Evening peak (5-7 PM)
    evening_mask = (hours >= 17) & (hours <= 19)
    base_concentration[evening_mask] = 7 + 12 * np.sin((hours[evening_mask] - 17) * np.pi / 2)
    
    # Add background level
    base_concentration += 2
    
    # Add noise
    concentration = base_concentration + np.random.normal(0, 1, len(times))
    concentration = np.maximum(concentration, 0)
    
    # Define risk levels
    low_risk_threshold = 5
    medium_risk_threshold = 15
    high_risk_threshold = 25
    
    # Risk classification
    risk_level = np.zeros(len(times), dtype=str)
    risk_level[(concentration < low_risk_threshold)] = 'Low'
    risk_level[(concentration >= low_risk_threshold) & (concentration < medium_risk_threshold)] = 'Medium'
    risk_level[(concentration >= medium_risk_threshold) & (concentration < high_risk_threshold)] = 'High'
    risk_level[(concentration >= high_risk_threshold)] = 'Very High'
    
    # Create dataframe
    data = pd.DataFrame({
        'Time': times,
        'Hour': hours,
        'Pathogen_Concentration': concentration,
        'Risk_Level': risk_level
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot concentration
    scatter = ax.scatter(data['Hour'], data['Pathogen_Concentration'], 
                        c=data['Risk_Level'].map({'Low': 'green', 'Medium': 'yellow', 
                                                 'High': 'orange', 'Very High': 'red'}), 
                        alpha=0.7, s=50)
    
    # Add risk level lines
    ax.axhline(y=low_risk_threshold, color='yellow', linestyle='--', alpha=0.7, label='Low Risk Threshold')
    ax.axhline(y=medium_risk_threshold, color='orange', linestyle='--', alpha=0.7, label='Medium Risk Threshold')
    ax.axhline(y=high_risk_threshold, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
    
    # Customize plot
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Pathogen Concentration (particles/m³)', fontsize=12)
    ax.set_title('AI-Driven Airborne Pathogen Detection', fontsize=14)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    legend_elements = [
        mpatches.Patch(color='green', label='Low Risk'),
        mpatches.Patch(color='yellow', label='Medium Risk'),
        mpatches.Patch(color='orange', label='High Risk'),
        mpatches.Patch(color='red', label='Very High Risk')
    ]
    ax.legend(handles=legend_elements, title='Risk Levels', loc='upper right')
    
    # Add metrics
    metrics_text = (
        "Detection Sensitivity: 94.2%\n"
        "Classification Accuracy: 92.8%\n"
        "Response Time: 30 seconds"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax.text(0.5, 25, metrics_text, fontsize=10, verticalalignment='bottom', bbox=props)
    
    # Annotate busy periods
    ax.annotate('Morning\nRush Hour', xy=(8, 15), xytext=(8, 20),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
    
    ax.annotate('Lunch\nHour', xy=(13, 14), xytext=(13, 19),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
    
    ax.annotate('Evening\nRush Hour', xy=(18, 19), xytext=(18, 24),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
    
    plt.tight_layout()
    save_plot(fig, 'figure8_airborne_pathogen_detection.png')
    return fig

# --------------------------------
# Figure 9: Real-Time Hospital Capacity Monitoring
# --------------------------------
def plot_hospital_capacity_monitoring():
    # Generate synthetic hospital capacity data
    np.random.seed(45)
    dates = pd.date_range(start='2023-05-01', end='2023-05-10')
    
    # Base capacity with weekend pattern
    weekday_effect = [0 if d.weekday() >= 5 else 1 for d in dates]  # Lower on weekends
    
    # ICU bed data
    icu_capacity = 120  # Total ICU beds
    base_icu_occupancy = np.array([65 + 20 * w for w in weekday_effect])
    icu_occupancy = base_icu_occupancy + np.random.normal(0, 5, len(dates))
    icu_occupancy = np.clip(icu_occupancy, 0, icu_capacity).astype(int)
    icu_occupancy_rate = icu_occupancy / icu_capacity * 100
    
    # Ventilator data
    ventilator_capacity = 80  # Total ventilators
    base_ventilator_usage = np.array([40 + 15 * w for w in weekday_effect])
    ventilator_usage = base_ventilator_usage + np.random.normal(0, 4, len(dates))
    ventilator_usage = np.clip(ventilator_usage, 0, ventilator_capacity).astype(int)
    ventilator_usage_rate = ventilator_usage / ventilator_capacity * 100
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'ICU_Beds_Total': icu_capacity,
        'ICU_Beds_Occupied': icu_occupancy,
        'ICU_Occupancy_Rate': icu_occupancy_rate,
        'Ventilators_Total': ventilator_capacity,
        'Ventilators_In_Use': ventilator_usage,
        'Ventilator_Usage_Rate': ventilator_usage_rate
    })
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot ICU occupancy
    ax1.bar(data['Date'], data['ICU_Beds_Occupied'], color='#3274A1', alpha=0.7, label='ICU Beds Occupied')
    ax1.bar(data['Date'], data['ICU_Beds_Total'] - data['ICU_Beds_Occupied'], 
           bottom=data['ICU_Beds_Occupied'], color='#E1812C', alpha=0.3, label='ICU Beds Available')
    
    # Plot critical threshold line
    critical_threshold = 0.8 * icu_capacity  # 80% occupancy is critical
    ax1.axhline(y=critical_threshold, color='red', linestyle='--', alpha=0.7, label='Critical Threshold (80%)')
    
    # Add percentage labels on bars
    for i, (date, occupied, rate) in enumerate(zip(data['Date'], data['ICU_Beds_Occupied'], data['ICU_Occupancy_Rate'])):
        ax1.text(date, occupied + 5, f"{rate:.1f}%", ha='center', fontsize=9)
    
    # Plot ventilator usage
    ax2.bar(data['Date'], data['Ventilators_In_Use'], color='#3274A1', alpha=0.7, label='Ventilators In Use')
    ax2.bar(data['Date'], data['Ventilators_Total'] - data['Ventilators_In_Use'], 
           bottom=data['Ventilators_In_Use'], color='#E1812C', alpha=0.3, label='Ventilators Available')
    
    # Add percentage labels on bars
    for i, (date, used, rate) in enumerate(zip(data['Date'], data['Ventilators_In_Use'], data['Ventilator_Usage_Rate'])):
        ax2.text(date, used + 3, f"{rate:.1f}%", ha='center', fontsize=9)
    
    # Customize first plot
    ax1.set_ylabel('ICU Beds')
    ax1.set_ylim(0, icu_capacity * 1.1)
    ax1.legend(loc='upper right')
    ax1.set_title('Real-Time Hospital and Healthcare Capacity Monitoring', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Customize second plot
    ax2.set_ylabel('Ventilators')
    ax2.set_ylim(0, ventilator_capacity * 1.1)
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    # Add metrics as text
    avg_icu_rate = data['ICU_Occupancy_Rate'].mean()
    avg_ventilator_rate = data['Ventilator_Usage_Rate'].mean()
    
    metrics_text = (
        f"Avg. ICU Bed Occupancy Rate: {avg_icu_rate:.1f}%\n"
        f"Avg. Ventilator Usage Rate: {avg_ventilator_rate:.1f}%\n"
        f"Processing Time: 85ms"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax1.text(dates[1], icu_capacity * 0.4, metrics_text, fontsize=10, 
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    save_plot(fig, 'figure9_hospital_capacity.png')
    return fig

# --------------------------------
# Figure 10: AI-Powered Predictive Models for Disease Spread
# --------------------------------
def plot_disease_spread_prediction():
    # Generate synthetic disease spread data
    np.random.seed(46)
    dates = pd.date_range(start='2023-01-01', end='2023-03-31')
    
    # Generate actual cases with exponential growth and then decline
    peak_day = 45
    actual_cases = []
    
    for i in range(len(dates)):
        if i < peak_day:
            # Growth phase
            base = 10 * np.exp(0.1 * i)
        else:
            # Decline phase
            base = 10 * np.exp(0.1 * peak_day) * np.exp(-0.05 * (i - peak_day))
        
        # Add weekly pattern
        weekday_effect = 1 - 0.2 * (dates[i].weekday() >= 5)  # Lower on weekends
        
        # Add noise
        case = base * weekday_effect * (1 + np.random.normal(0, 0.1))
        actual_cases.append(int(case))
    
    # Generate predictions (made 7 days earlier)
    predicted_cases = []
    prediction_intervals_lower = []
    prediction_intervals_upper = []
    
    for i in range(len(dates)):
        if i < 7:
            # No predictions for first week
            predicted = np.nan
            lower = np.nan
            upper = np.nan
        else:
            # Prediction with some error
            error_factor = 1 + np.random.normal(0, 0.1)
            predicted = actual_cases[i - 7] * np.exp(0.1 * 7) * error_factor
            
            # Prediction intervals
            lower = predicted * 0.8
            upper = predicted * 1.2
        
        predicted_cases.append(int(predicted) if not np.isnan(predicted) else np.nan)
        prediction_intervals_lower.append(lower)
        prediction_intervals_upper.append(upper)
    
    # Create dataframe
    data = pd.DataFrame({
        'Date': dates,
        'Actual_Cases': actual_cases,
        'Predicted_Cases': predicted_cases,
        'Lower_Bound': prediction_intervals_lower,
        'Upper_Bound': prediction_intervals_upper
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual and predicted cases
    ax.plot(data['Date'], data['Actual_Cases'], 'b-', label='Actual Cases', linewidth=2)
    ax.plot(data['Date'], data['Predicted_Cases'], 'r--', label='Predicted Cases', linewidth=2)
    
    # Plot prediction intervals
    ax.fill_between(data['Date'], data['Lower_Bound'], data['Upper_Bound'], 
                   color='red', alpha=0.2, label='Prediction Interval')
    
    # Calculate error metrics
    valid_indices = ~np.isnan(data['Predicted_Cases'])
    
    if np.any(valid_indices):
        actual = np.array(data.loc[valid_indices, 'Actual_Cases'])
        predicted = np.array(data.loc[valid_indices, 'Predicted_Cases'])
        
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Accuracy based on whether predictions are within 20% of actual values
        within_range = np.abs((actual - predicted) / actual) <= 0.2
        accuracy = np.mean(within_range) * 100
    else:
        mae = np.nan
        mape = np.nan
        accuracy = np.nan
    
    # Customize plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily New Cases', fontsize=12)
    ax.set_title('AI-Powered Predictive Models for Disease Spread', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add metrics as text
    metrics_text = (
        f"Forecasting Accuracy: {accuracy:.1f}%\n"
        f"Mean Absolute Error: {mae:.1f}\n"
        f"Mean Absolute Percentage Error: {mape:.1f}%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax.text(dates[60], np.max(actual_cases) * 0.8, metrics_text, fontsize=10, 
             verticalalignment='bottom', bbox=props)
    
    # Annotate key events
    ax.annotate('Peak of\nOutbreak', xy=(dates[peak_day], actual_cases[peak_day]), 
               xytext=(dates[peak_day-15], actual_cases[peak_day] * 1.2),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
    
    plt.tight_layout()
    save_plot(fig, 'figure10_disease_spread_prediction.png')
    return fig

# --------------------------------
# Figure 11: AI-Powered Smart Pharmacies and Medicine Demand Prediction
# --------------------------------
def plot_medicine_demand_prediction():
    # Pull relevant medication data from pharmaceutical sales dataset
    tamiflu_data = pharma_df[pharma_df['Medication'] == 'Tamiflu'].copy()
    paracetamol_data = pharma_df[pharma_df['Medication'] == 'Paracetamol'].copy()
    
    # Ensure sorted by date
    tamiflu_data = tamiflu_data.sort_values('Date')
    paracetamol_data = paracetamol_data.sort_values('Date')
    
    # Calculate rolling averages for smoothing
    tamiflu_rolling = tamiflu_data['Units_Sold'].rolling(window=7, min_periods=1).mean()
    paracetamol_rolling = paracetamol_data['Units_Sold'].rolling(window=7, min_periods=1).mean()
    
    # Find baseline (normal demand)
    tamiflu_baseline = np.percentile(tamiflu_data['Units_Sold'], 25)
    paracetamol_baseline = np.percentile(paracetamol_data['Units_Sold'], 25)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot medication sales
    ax.plot(tamiflu_data['Date'], tamiflu_rolling, 'b-', label='Tamiflu (7-day avg)', linewidth=2)
    ax.plot(paracetamol_data['Date'], paracetamol_rolling, 'r-', label='Paracetamol (7-day avg)', linewidth=2)
    
    # Add horizontal lines for baselines
    ax.axhline(y=tamiflu_baseline, color='blue', linestyle='--', alpha=0.5, label='Tamiflu Baseline')
    ax.axhline(y=paracetamol_baseline, color='red', linestyle='--', alpha=0.5, label='Paracetamol Baseline')
    
    # Highlight outbreak periods
    outbreak_dates = [pd.to_datetime('2023-01-15'), pd.to_datetime('2023-04-10'), 
                     pd.to_datetime('2023-07-20'), pd.to_datetime('2023-10-05')]
    
    for outbreak_date in outbreak_dates:
        start_date = outbreak_date - pd.Timedelta(days=5)
        end_date = outbreak_date + pd.Timedelta(days=14)
        ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
        ax.text(outbreak_date, ax.get_ylim()[1] * 0.95, 'Outbreak', 
                rotation=90, verticalalignment='top', fontsize=10)
    
    # Calculate percent increase during outbreaks
    tamiflu_max = tamiflu_data[(tamiflu_data['Date'] >= outbreak_dates[0] - pd.Timedelta(days=5)) & 
                             (tamiflu_data['Date'] <= outbreak_dates[0] + pd.Timedelta(days=14))]['Units_Sold'].max()
    paracetamol_max = paracetamol_data[(paracetamol_data['Date'] >= outbreak_dates[0] - pd.Timedelta(days=5)) & 
                                    (paracetamol_data['Date'] <= outbreak_dates[0] + pd.Timedelta(days=14))]['Units_Sold'].max()
    
    tamiflu_percent_increase = (tamiflu_max / tamiflu_baseline - 1) * 100
    paracetamol_percent_increase = (paracetamol_max / paracetamol_baseline - 1) * 100
    
    # Customize plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Units Sold', fontsize=12)
    ax.set_title('AI-Powered Smart Pharmacies and Medicine Demand Prediction', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add metrics as text
    metrics_text = (
        f"Tamiflu Increase: {tamiflu_percent_increase:.1f}%\n"
        f"Paracetamol Increase: {paracetamol_percent_increase:.1f}%\n"
        f"Early Warning Time: 3-5 days before outbreak peak"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax.text(pd.to_datetime('2023-02-15'), ax.get_ylim()[1] * 0.5, metrics_text, fontsize=10, 
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    save_plot(fig, 'figure11_medicine_demand_prediction.png')
    return fig

# --------------------------------
# Figure 12: AI-Powered Wastewater Surveillance
# --------------------------------
def plot_wastewater_surveillance():
    # Filter wastewater data for SARS-CoV-2
    covid_data = wastewater_df[wastewater_df['Pathogen'] == 'SARS-CoV-2'].copy()
    
    # Create pivot table for location and date
    pivot_data = covid_data.pivot_table(index='Date', columns='Location', 
                                       values='Concentration_copies_per_mL', aggfunc='mean')
    
    # Plot concentration over time by location
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each location
    for location in pivot_data.columns:
        ax.plot(pivot_data.index, pivot_data[location], label=location, linewidth=2, alpha=0.7)
    
    # Highlight outbreak periods
    outbreak_dates = [pd.to_datetime('2023-01-15'), pd.to_datetime('2023-04-10'), 
                     pd.to_datetime('2023-07-20'), pd.to_datetime('2023-10-05')]
    
    for outbreak_date in outbreak_dates:
        start_date = outbreak_date - pd.Timedelta(days=7)
        end_date = outbreak_date + pd.Timedelta(days=7)
        ax.axvspan(start_date, end_date, color='red', alpha=0.2)
        ax.text(outbreak_date, ax.get_ylim()[1] * 0.95, 'Outbreak', 
                rotation=90, verticalalignment='top', fontsize=10, color='darkred')
    
    # Customize plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('SARS-CoV-2 Concentration (copies/mL)', fontsize=12)
    ax.set_title('AI-Powered Wastewater Surveillance for Pathogen Detection', fontsize=14)
    ax.legend(title='Location', loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add metrics
    metrics_text = (
        "Detection Sensitivity: 91.6%\n"
        "False Positive Rate: 6.8%\n"
        "Pathogen Classification Accuracy: 94.1%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax.text(pd.to_datetime('2023-02-15'), ax.get_ylim()[1] * 0.5, metrics_text, fontsize=10, 
            verticalalignment='bottom', bbox=props)
    
    # Annotate lead time before clinical cases
    ax.annotate('7-10 Day Lead Time\nBefore Clinical Cases', 
               xy=(outbreak_dates[0] - pd.Timedelta(days=7), ax.get_ylim()[1] * 0.7),
               xytext=(outbreak_dates[0] - pd.Timedelta(days=20), ax.get_ylim()[1] * 0.8),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
    
    plt.tight_layout()
    save_plot(fig, 'figure12_wastewater_surveillance.png')
    return fig

# --------------------------------
# Figure 13: AI Effectiveness in Real-Time Monitoring
# --------------------------------
def plot_ai_effectiveness():
    # Effectiveness data from the paper
    methods = ['Live Dashboards', 'Thermal Imaging', 'Social Media Mining']
    effectiveness = [85, 75, 90]
    
    # Create more detailed data for radar chart
    categories = ['Accuracy', 'Speed', 'Cost Efficiency', 'Scalability', 'Early Detection']
    
    # Values for each method across categories (0-10 scale)
    dashboard_scores = [8.5, 9.0, 7.0, 8.5, 8.0]
    thermal_scores = [7.5, 8.0, 6.0, 5.5, 9.0]
    social_scores = [9.0, 9.5, 8.5, 10.0, 8.0]
    
    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Bar chart for overall effectiveness
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    bars = ax1.bar(methods, effectiveness, color=['#3274A1', '#E1812C', '#3A923A'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Effectiveness Score (%)')
    ax1.set_title('AI Effectiveness in Real-Time Disease Monitoring', fontsize=16, pad=20)
    
    # Radar chart
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2, polar=True)
    
    # Number of categories
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the category labels
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    
    # Draw the chart for each method
    # Add the first method
    dashboard_scores += dashboard_scores[:1]  # Close the loop
    ax2.plot(angles, dashboard_scores, linewidth=2, linestyle='solid', label='Live Dashboards')
    ax2.fill(angles, dashboard_scores, alpha=0.1)
    
    # Add the second method
    thermal_scores += thermal_scores[:1]  # Close the loop
    ax2.plot(angles, thermal_scores, linewidth=2, linestyle='solid', label='Thermal Imaging')
    ax2.fill(angles, thermal_scores, alpha=0.1)
    
    # Add the third method
    social_scores += social_scores[:1]  # Close the loop
    ax2.plot(angles, social_scores, linewidth=2, linestyle='solid', label='Social Media Mining')
    ax2.fill(angles, social_scores, alpha=0.1)
    
    # Set y-limits
    ax2.set_ylim(0, 10)
    ax2.set_yticks([2, 4, 6, 8, 10])
    ax2.set_yticklabels(['2', '4', '6', '8', '10'])
    
    # Add legend
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Add title for radar chart
    ax2.set_title('Performance Metrics by Monitoring Method', fontsize=14, pad=20)
    
    plt.tight_layout()
    save_plot(fig, 'figure13_ai_effectiveness.png')
    return fig

# --------------------------------
# Figure 14: Wearable Devices Detecting Abnormal Health Metrics
# --------------------------------
def plot_wearable_abnormal_metrics():
    # Extract data from wearable dataset
    normal_temp = wearable_df[wearable_df['Fever_Detected'] == False]['Body_Temperature']
    fever_temp = wearable_df[wearable_df['Fever_Detected'] == True]['Body_Temperature']
    
    normal_hr = wearable_df[wearable_df['Fever_Detected'] == False]['Heart_Rate']
    fever_hr = wearable_df[wearable_df['Fever_Detected'] == True]['Heart_Rate']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot temperature distributions
    sns.kdeplot(data=normal_temp, ax=ax1, label='Normal', color='blue', fill=True)
    sns.kdeplot(data=fever_temp, ax=ax1, label='Fever Detected', color='red', fill=True)
    
    # Plot heart rate distributions
    sns.kdeplot(data=normal_hr, ax=ax2, label='Normal', color='blue', fill=True)
    sns.kdeplot(data=fever_hr, ax=ax2, label='Fever Detected', color='red', fill=True)
    
    # Add vertical lines for thresholds
    ax1.axvline(x=38.0, color='black', linestyle='--', label='Fever Threshold (38°C)')
    ax2.axvline(x=100, color='black', linestyle='--', label='Elevated HR Threshold (100 BPM)')
    
    # Customize plots
    ax1.set_xlabel('Body Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Body Temperature Distribution', fontsize=14)
    ax1.legend()
    
    ax2.set_xlabel('Heart Rate (BPM)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Heart Rate Distribution', fontsize=14)
    ax2.legend()
    
    # Add metrics as text
    metrics_text = (
        "Fever Detection Accuracy: 93.5%\n"
        "Anomalous Heart Rate Detection: 91.8%\n"
        "False Positive Rate: 5.4%\n"
        "False Negative Rate: 6.1%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    ax1.text(36, 1.0, metrics_text, fontsize=10, verticalalignment='bottom', bbox=props)
    
    plt.suptitle('Wearable Devices Detecting Abnormal Health Metrics', fontsize=16, y=1.05)
    plt.tight_layout()
    save_plot(fig, 'figure14_wearable_abnormal_metrics.png')
    return fig

# Generate all plots
def generate_all_plots():
    """Generate all plots from the paper"""
    print("Generating all plots...")
    
    print("Figure 2: Performance Evaluation")
    plot_performance_evaluation()
    
    print("Figure 3: Confusion Matrix")
    plot_confusion_matrix()
    
    print("Figure 4: Outbreak Comparison")
    plot_outbreak_comparison()
    
    print("Figure 5: Wearable Monitoring")
    plot_wearable_monitoring()
    
    print("Figure 6: Early Outbreak Detection")
    plot_early_outbreak_detection()
    
    print("Figure 7: Contact Tracing")
    plot_contact_tracing()
    
    print("Figure 8: Airborne Pathogen Detection")
    plot_airborne_pathogen_detection()
    
    print("Figure 9: Hospital Capacity")
    plot_hospital_capacity_monitoring()
    
    print("Figure 10: Disease Spread Prediction")
    plot_disease_spread_prediction()
    
    print("Figure 11: Medicine Demand Prediction")
    plot_medicine_demand_prediction()
    
    print("Figure 12: Wastewater Surveillance")
    plot_wastewater_surveillance()
    
    print("Figure 13: AI Effectiveness")
    plot_ai_effectiveness()
    
    print("Figure 14: Wearable Abnormal Metrics")
    plot_wearable_abnormal_metrics()
    
    print("All plots generated and saved to 'plots' directory.")

if __name__ == "__main__":
    generate_all_plots()
