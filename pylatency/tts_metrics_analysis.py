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
from loguru import logger
import sys
import argparse

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Set the style for all visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='TTS Metrics Analysis Script - Generates visualizations and statistics for TTS services.'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input metrics CSV file path. If not provided, uses latest metrics file from data directory.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory path for analysis results. Default: ./data/metrics_analysis_output',
        default='./data/metrics_analysis_output'
    )
    return parser.parse_args()

def load_metrics_data(file_path: Path) -> pd.DataFrame:
    """Load and preprocess metrics data from CSV file."""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # logger.info column names for debugging
    logger.info("Available columns:", df.columns.tolist())
    
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
        'GoAzure': ['#00A4EF', '#69D1FF'],     # Microsoft blue shades
        'Azure': ['#00A4EF', '#69D1FF'],        # Microsoft blue shades
        'AzureV2': ['#0078D4', '#50B0FF'],      # Different Microsoft blue shades
        'Deepgram': ['#00C805', '#7DFF82'],     # Vibrant green shades
        'OpenAI': ['#FF3366', '#FF99B3']        # Bright pink shades
    }
    
    # Get unique services
    services = df['service'].unique()
    logger.info(f"Found services: {services}")
    
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
    logger.info(f"Saved combined plot to {output_file}")

def plot_latency_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot latency distribution for each service separately."""
    services = df['service'].unique()
    
    # Define vibrant colors for each service
    service_colors = {
        'GoAzure': '#00A4EF',     # Microsoft blue
        'Azure': '#00A4EF',        # Microsoft blue
        'AzureV2': '#0078D4',      # Different Microsoft blue
        'Deepgram': '#00C805',     # Vibrant green
        'OpenAI': '#FF3366'        # Bright pink
    }
    
    for service in services:
        service_df = df[df['service'] == service].copy()
        
        # Drop any NaN values
        service_df = service_df.dropna(subset=['ttfb_ms', 'e2e_latency_ms'])
        
        plt.figure(figsize=(12, 6))
        
        # Create violin plots for TTFB and E2E latency
        data = [
            service_df['ttfb_ms'].values,  # Explicitly get values
            service_df['e2e_latency_ms'].values  # Explicitly get values
        ]
        
        # Print debug info
        logger.debug(f"{service} TTFB stats: min={service_df['ttfb_ms'].min():.1f}, max={service_df['ttfb_ms'].max():.1f}")
        logger.debug(f"{service} E2E stats: min={service_df['e2e_latency_ms'].min():.1f}, max={service_df['e2e_latency_ms'].max():.1f}")
        
        labels = ['TTFB', 'E2E Latency']
        
        # Create violin plots with wider range and ensure data is plotted
        violin_parts = plt.violinplot(data, showmedians=True, points=100, widths=0.7)
        
        # Customize violin plot colors and transparency
        for pc in violin_parts['bodies']:
            pc.set_facecolor(service_colors[service])
            pc.set_alpha(0.7)
        
        # Add box plots inside violin plots with outliers
        box_parts = plt.boxplot(data, widths=0.2, showfliers=True, positions=[1,2])
        
        # Calculate and show statistics
        ttfb_median = service_df['ttfb_ms'].median()
        e2e_median = service_df['e2e_latency_ms'].median()
        ttfb_95th = service_df['ttfb_ms'].quantile(0.95)
        e2e_95th = service_df['e2e_latency_ms'].quantile(0.95)
        
        # Add statistics annotations with adjusted positions
        plt.annotate(f'Median: {ttfb_median:.1f}ms\n95th: {ttfb_95th:.1f}ms', 
                    xy=(1, ttfb_median), xytext=(0.6, ttfb_median),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate(f'Median: {e2e_median:.1f}ms\n95th: {e2e_95th:.1f}ms', 
                    xy=(2, e2e_median), xytext=(2.4, e2e_median),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.title(f'{service} TTS Latency Distribution')
        plt.xticks([1, 2], labels)
        plt.ylabel('Latency (ms)')
        plt.grid(True, alpha=0.3)
        
        # Ensure y-axis shows full range with some padding
        max_latency = max(
            service_df['e2e_latency_ms'].max(),
            service_df['ttfb_ms'].max()
        )
        plt.ylim(0, max_latency * 1.2)  # 20% padding
        
        # Save the plot
        plt.savefig(output_dir / f'latency_distribution_{service.lower()}.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_service_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot service comparison metrics."""
    plt.figure(figsize=(14, 6))  # Increased width to accommodate more services
    
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
    
    # Sort services to group Azure services together
    services = sorted(service_metrics.index, key=lambda x: (
        0 if x.startswith('Azure') else 1,  # Azure services first
        x  # Then alphabetically
    ))
    service_metrics = service_metrics.reindex(services)
    
    # Prepare data for plotting
    x = np.arange(len(services))
    width = 0.35
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
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
            '95th Percentile TTFB (ms)': service_df['ttfb_ms'].quantile(0.95),
            '99th Percentile TTFB (ms)': service_df['ttfb_ms'].quantile(0.99),
            'Mean E2E (ms)': service_df['e2e_latency_ms'].mean(),
            'Median E2E (ms)': service_df['e2e_latency_ms'].median(),
            'Std E2E (ms)': service_df['e2e_latency_ms'].std(),
            '95th Percentile E2E (ms)': service_df['e2e_latency_ms'].quantile(0.95),
            '99th Percentile E2E (ms)': service_df['e2e_latency_ms'].quantile(0.99),
            'Processing Time (ms)': (service_df['e2e_latency_ms'] - service_df['ttfb_ms']).mean(),
            'Sample Size': len(service_df)
        }
        stats.append(service_stats)
    
    # Round all numeric values to 2 decimal places
    stats_df = pd.DataFrame(stats)
    numeric_columns = stats_df.select_dtypes(include=['float64']).columns
    stats_df[numeric_columns] = stats_df[numeric_columns].round(2)
    
    return stats_df

def get_latest_metrics_file(data_dir: Path) -> Path:
    """Get the most recent metrics CSV file from the data directory."""
    metrics_files = list(data_dir.glob("*metrics*.csv"))
    if not metrics_files:
        raise FileNotFoundError("No metrics files found in data directory")
    
    # Sort by modification time, most recent first
    latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using metrics file: {latest_file}")
    return latest_file

def main():
    """Main function to run the analysis."""
    args = parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Get metrics file
        if args.input:
            metrics_file = Path(args.input)
            if not metrics_file.exists():
                raise FileNotFoundError(f"Input file not found: {metrics_file}")
        else:
            data_dir = script_dir / "data"
            metrics_file = get_latest_metrics_file(data_dir)
        
        logger.info(f"Using metrics file: {metrics_file}")
        
        # Load and validate data
        df = load_metrics_data(metrics_file)
        logger.info(f"Loaded {len(df)} records")
        
        # Generate visualizations
        plot_latency_over_time(df, output_dir)
        plot_latency_distribution(df, output_dir)
        plot_service_comparison(df, output_dir)
        
        # Calculate and save statistics
        stats_df = calculate_statistics(df)
        stats_file = output_dir / 'service_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"\nService Statistics saved to {stats_file}")
        logger.info(stats_df.to_string())
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 