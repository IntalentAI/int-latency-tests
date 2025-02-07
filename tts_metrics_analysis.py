#!/usr/bin/env python3
"""
TTS Metrics Analysis Script
Generates comprehensive visualizations and statistics for comparing different TTS services.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import matplotlib.dates as mdates

# Set the style for all visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_metrics_data(file_path: Path) -> pd.DataFrame:
    """Load and preprocess metrics data from CSV file."""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Map expected column names to actual column names
    column_mapping = {
        'Service': 'service',
        'Start Time': 'start_time',
        'TTFB (ms)': 'ttfb_ms',
        'E2E Latency (ms)': 'e2e_latency_ms'
    }
    
    # Rename columns if needed
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_columns = ['service', 'start_time', 'ttfb_ms', 'e2e_latency_ms']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df

def plot_latency_over_time(df: pd.DataFrame, output_dir: Path):
    """Plot TTFB and E2E latency over time for all services in one plot."""
    plt.figure(figsize=(15, 8))
    
    # Define colors for each service - vibrant, modern palette
    colors = {
        'Azure': ['#00A4EF', '#69D1FF'],     # Microsoft blue shades
        'Deepgram': ['#00C805', '#7DFF82'],  # Vibrant green shades
        'OpenAI': ['#FF3366', '#FF99B3']     # Bright pink shades
    }
    
    # Get unique services
    services = df['service'].unique()
    print(f"Found services: {services}")
    
    # Create one plot with all services
    for service in services:
        service_df = df[df['service'] == service].copy()
        
        # Convert timestamps to datetime
        try:
            service_df['start_time'] = pd.to_datetime(service_df['start_time'], unit='s')
        except (ValueError, TypeError):
            service_df['start_time'] = pd.to_datetime(service_df['start_time'])
        
        # Plot TTFB and E2E latency for this service
        plt.plot(service_df['start_time'], service_df['ttfb_ms'],
                marker='o', label=f'{service} TTFB',
                color=colors[service][0], linestyle='-', markersize=4)
        plt.plot(service_df['start_time'], service_df['e2e_latency_ms'],
                marker='s', label=f'{service} E2E',
                color=colors[service][1], linestyle='--', markersize=4)
    
    # Format x-axis to show time in HH:MM:SS
    time_fmt = mdates.DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(time_fmt)
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    
    plt.title('TTS Latency Over Time - All Services')
    plt.xlabel('Start Time')
    plt.ylabel('Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the combined plot
    output_file = output_dir / 'latency_over_time_combined.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved combined plot to {output_file}")

def plot_latency_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot latency distribution for each service separately."""
    services = df['service'].unique()
    
    # Define vibrant colors for each service
    service_colors = {
        'Azure': '#00A4EF',     # Bright blue
        'Deepgram': '#00C805',  # Vibrant green
        'OpenAI': '#FF3366'     # Bright pink
    }
    
    for service in services:
        service_df = df[df['service'] == service]
        
        plt.figure(figsize=(12, 6))
        
        # Create violin plots for TTFB and E2E latency
        data = [service_df['ttfb_ms'], service_df['e2e_latency_ms']]
        labels = ['TTFB', 'E2E Latency']
        
        violin_parts = plt.violinplot(data, showmedians=True)
        
        # Customize violin plot colors
        for pc in violin_parts['bodies']:
            pc.set_facecolor(service_colors[service])
            pc.set_alpha(0.7)
        
        # Add box plots inside violin plots
        plt.boxplot(data, widths=0.2)
        
        plt.title(f'{service} TTS Latency Distribution')
        plt.xticks([1, 2], labels)
        plt.ylabel('Latency (ms)')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(output_dir / f'latency_distribution_{service.lower()}.png')
        plt.close()

def plot_service_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot service comparison metrics."""
    plt.figure(figsize=(12, 6))
    
    # Define vibrant colors for bars
    bar_colors = {
        'ttfb': '#00A4EF',    # Bright blue
        'e2e': '#FF3366'      # Bright pink
    }
    
    # Calculate mean latencies for each service
    service_metrics = df.groupby('service').agg({
        'ttfb_ms': ['mean', 'std'],
        'e2e_latency_ms': ['mean', 'std']
    }).round(2)
    
    # Prepare data for plotting
    services = service_metrics.index
    x = np.arange(len(services))
    width = 0.35
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ttfb_bars = ax.bar(x - width/2, service_metrics['ttfb_ms']['mean'], 
                       width, label='TTFB', color=bar_colors['ttfb'])
    e2e_bars = ax.bar(x + width/2, service_metrics['e2e_latency_ms']['mean'], 
                      width, label='E2E Latency', color=bar_colors['e2e'])
    
    # Add error bars
    ax.errorbar(x - width/2, service_metrics['ttfb_ms']['mean'],
               yerr=service_metrics['ttfb_ms']['std'],
               fmt='none', color='black', capsize=5)
    ax.errorbar(x + width/2, service_metrics['e2e_latency_ms']['mean'],
               yerr=service_metrics['e2e_latency_ms']['std'],
               fmt='none', color='black', capsize=5)
    
    # Customize plot
    ax.set_ylabel('Latency (ms)')
    ax.set_title('TTS Service Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(services)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}ms',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
    
    autolabel(ttfb_bars)
    autolabel(e2e_bars)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'service_comparison.png')
    plt.close()

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive statistics for each service."""
    stats = []
    for service in df['service'].unique():
        service_df = df[df['service'] == service]
        
        service_stats = {
            'Service': service,
            'Mean TTFB (ms)': service_df['ttfb_ms'].mean(),
            'Median TTFB (ms)': service_df['ttfb_ms'].median(),
            'Std TTFB (ms)': service_df['ttfb_ms'].std(),
            'Mean E2E (ms)': service_df['e2e_latency_ms'].mean(),
            'Median E2E (ms)': service_df['e2e_latency_ms'].median(),
            'Std E2E (ms)': service_df['e2e_latency_ms'].std(),
            '95th Percentile E2E (ms)': service_df['e2e_latency_ms'].quantile(0.95),
            'Processing Time (ms)': (service_df['e2e_latency_ms'] - service_df['ttfb_ms']).mean(),
            'Sample Size': len(service_df)
        }
        stats.append(service_stats)
    
    return pd.DataFrame(stats)

def get_latest_metrics_file(data_dir: Path) -> Path:
    """Get the most recent metrics CSV file from the data directory."""
    metrics_files = list(data_dir.glob("*metrics*.csv"))
    if not metrics_files:
        raise FileNotFoundError("No metrics files found in data directory")
    
    # Sort by modification time, most recent first
    latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
    print(f"Using metrics file: {latest_file}")
    return latest_file

def main():
    """Main function to run the analysis."""
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_dir = data_dir / "metrics_analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    # Get latest metrics file
    try:
        metrics_file = get_latest_metrics_file(data_dir)
        print(f"Using metrics file: {metrics_file}")
        
        # Load and validate data
        df = load_metrics_data(metrics_file)
        print(f"Loaded {len(df)} records")
        
        # Generate visualizations
        plot_latency_over_time(df, output_dir)
        plot_latency_distribution(df, output_dir)
        plot_service_comparison(df, output_dir)
        
        # Calculate and save statistics
        stats_df = calculate_statistics(df)
        stats_file = output_dir / 'service_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        print(f"\nService Statistics saved to {stats_file}")
        print(stats_df.to_string())
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 