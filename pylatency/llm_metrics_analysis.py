#!/usr/bin/env python3
"""
LLM Metrics Analysis Script
Generates comprehensive visualizations and statistics for comparing LLM performance across different regions and services,
with specific focus on conversation-based metrics.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import matplotlib.dates as mdates
from loguru import logger
import sys
import argparse
import json
from collections import defaultdict
import os

# Create base output directory
base_output_dir = "llm_metrics_analysis_output"
os.makedirs(base_output_dir, exist_ok=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger.add(
    f"{base_output_dir}/analysis_log_{timestamp}.log",
    level="DEBUG",
)

# Set the style for all visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LLM Metrics Analysis Script - Generates visualizations and statistics for conversation-based LLM services.'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input metrics CSV file path. If not provided, uses latest metrics file from llm_metrics_analysis_output directory.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory path for analysis results. Default: ./llm_metrics_analysis_output/analysis_[timestamp]',
        default=os.path.join(base_output_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Filter analysis to specific model (e.g., gpt-4, gpt-35-turbo)'
    )
    parser.add_argument(
        '--responses-dir',
        type=str,
        help='Directory containing saved response JSON files for detailed conversation analysis'
    )
    return parser.parse_args()

def load_metrics_data(file_path: Path) -> pd.DataFrame:
    """Load and preprocess metrics data from CSV file."""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Log column names for debugging
    logger.info(f"Available columns: {df.columns.tolist()}")
    
    # Map expected column names to actual column names
    column_mapping = {
        'Service': 'service',
        'Region': 'region',
        'Model': 'model',
        'Input Text': 'input_text',
        'TTFT (ms)': 'ttft_ms',
        'E2E Latency (ms)': 'e2e_latency_ms',
        'Tokens/Second': 'tokens_per_second',
        'Input Tokens': 'input_tokens',
        'Output Tokens': 'output_tokens',
        'Start Time': 'start_time',
        'End Time': 'end_time'
    }
    
    # Rename columns if needed
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_columns = ['service', 'region', 'model', 'ttft_ms', 'e2e_latency_ms', 'tokens_per_second', 'input_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert timestamps to datetime if they're not already
    for col in ['start_time', 'end_time']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], unit='s')
            except (ValueError, TypeError):
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    logger.warning(f"Could not convert {col} to datetime")
    
    # Add derived metrics for conversation analysis
    df['conversation_length'] = df['input_text'].str.len()
    df['tokens_per_char'] = df['input_tokens'] / df['conversation_length']
    
    return df

def plot_latency_over_time(df: pd.DataFrame, output_dir: Path, model_filter: Optional[str] = None):
    """Plot TTFT and E2E latency over time for all regions/services."""
    if model_filter:
        df = df[df['model'] == model_filter]
        title_suffix = f" - {model_filter}"
    else:
        title_suffix = ""
    
    plt.figure(figsize=(15, 8))
    
    # Define colors for different regions - vibrant, modern palette
    colors = {
        'eastus': ['#FF3366', '#FF99B3'],      # Pink shades
        'westus': ['#00A4EF', '#69D1FF'],      # Blue shades
        'westeurope': ['#7FBA00', '#A2D45E'],  # Green shades
        'southcentralus': ['#FFB900', '#FFDB70'], # Yellow shades
        'northeurope': ['#737373', '#B3B3B3'],  # Gray shades
        # Add more regions as needed
    }
    
    # Default colors for regions not in the predefined list
    default_colors = ['#8661C5', '#B292DC']  # Purple shades
    
    # Get unique region/service combinations
    df['region_service'] = df['region'] + ' - ' + df['service']
    region_services = df['region_service'].unique()
    logger.info(f"Found region-service combinations: {region_services}")
    
    # Create one plot with all regions
    for region_service in region_services:
        region = region_service.split(' - ')[0]
        service = region_service.split(' - ')[1]
        
        region_df = df[df['region_service'] == region_service].copy()
        
        # Use predefined colors or default colors
        color_pair = colors.get(region, default_colors)
        
        # Plot TTFT and E2E latency for this region
        plt.plot(region_df['start_time'], region_df['ttft_ms'],
                marker='o', label=f'{region_service} TTFT',
                color=color_pair[0], linestyle='-', markersize=4)
        plt.plot(region_df['start_time'], region_df['e2e_latency_ms'],
                marker='s', label=f'{region_service} E2E',
                color=color_pair[1], linestyle='--', markersize=4)
    
    # Format x-axis to show time in HH:MM:SS
    time_fmt = mdates.DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(time_fmt)
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    
    plt.title(f'LLM Latency Over Time{title_suffix}')
    plt.xlabel('Start Time')
    plt.ylabel('Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the combined plot
    model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
    output_file = output_dir / f'latency_over_time{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved latency over time plot to {output_file}")

def plot_tokens_per_second(df: pd.DataFrame, output_dir: Path, model_filter: Optional[str] = None):
    """Plot tokens per second over time for all regions/services."""
    if model_filter:
        df = df[df['model'] == model_filter]
        title_suffix = f" - {model_filter}"
    else:
        title_suffix = ""
    
    plt.figure(figsize=(15, 8))
    
    # Define colors for different regions
    colors = {
        'eastus': '#FF3366',      # Pink
        'westus': '#00A4EF',      # Blue
        'westeurope': '#7FBA00',  # Green
        'southcentralus': '#FFB900', # Yellow
        'northeurope': '#737373',  # Gray
    }
    
    # Default color for regions not in the predefined list
    default_color = '#8661C5'  # Purple
    
    # Get unique region/service combinations
    df['region_service'] = df['region'] + ' - ' + df['service']
    region_services = df['region_service'].unique()
    
    # Create one plot with all regions
    for region_service in region_services:
        region = region_service.split(' - ')[0]
        
        region_df = df[df['region_service'] == region_service].copy()
        
        # Use predefined color or default color
        color = colors.get(region, default_color)
        
        # Plot tokens per second for this region
        plt.plot(region_df['start_time'], region_df['tokens_per_second'],
                marker='o', label=f'{region_service}',
                color=color, linestyle='-', markersize=4)
    
    # Format x-axis to show time in HH:MM:SS
    time_fmt = mdates.DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(time_fmt)
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    
    plt.title(f'LLM Throughput (Tokens/Second) Over Time{title_suffix}')
    plt.xlabel('Start Time')
    plt.ylabel('Tokens per Second')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
    output_file = output_dir / f'tokens_per_second{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved tokens per second plot to {output_file}")

def plot_latency_distribution(df: pd.DataFrame, output_dir: Path, model_filter: Optional[str] = None):
    """Plot latency distribution for each region separately."""
    if model_filter:
        df = df[df['model'] == model_filter]
        title_suffix = f" - {model_filter}"
    else:
        title_suffix = ""
    
    # Get unique region/service combinations
    df['region_service'] = df['region'] + ' - ' + df['service']
    region_services = df['region_service'].unique()
    
    # Define vibrant colors for regions
    region_colors = {
        'eastus': '#FF3366',      # Pink
        'westus': '#00A4EF',      # Blue
        'westeurope': '#7FBA00',  # Green
        'southcentralus': '#FFB900', # Yellow
        'northeurope': '#737373',  # Gray
    }
    
    # Default color for regions not in the predefined list
    default_color = '#8661C5'  # Purple
    
    for region_service in region_services:
        region = region_service.split(' - ')[0]
        
        region_df = df[df['region_service'] == region_service].copy()
        
        # Drop any NaN values
        region_df = region_df.dropna(subset=['ttft_ms', 'e2e_latency_ms'])
        
        plt.figure(figsize=(12, 6))
        
        # Create violin plots for TTFT and E2E latency
        data = [
            region_df['ttft_ms'].values,
            region_df['e2e_latency_ms'].values
        ]
        
        # Print debug info
        logger.debug(f"{region_service} TTFT stats: min={region_df['ttft_ms'].min():.1f}, max={region_df['ttft_ms'].max():.1f}")
        logger.debug(f"{region_service} E2E stats: min={region_df['e2e_latency_ms'].min():.1f}, max={region_df['e2e_latency_ms'].max():.1f}")
        
        labels = ['TTFT', 'E2E Latency']
        
        # Use predefined color or default color
        color = region_colors.get(region, default_color)
        
        # Create violin plots
        violin_parts = plt.violinplot(data, showmedians=True, points=100, widths=0.7)
        
        # Customize violin plot colors and transparency
        for pc in violin_parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add box plots inside violin plots with outliers
        box_parts = plt.boxplot(data, widths=0.2, showfliers=True, positions=[1,2])
        
        # Calculate and show statistics
        ttft_median = region_df['ttft_ms'].median()
        e2e_median = region_df['e2e_latency_ms'].median()
        ttft_95th = region_df['ttft_ms'].quantile(0.95)
        e2e_95th = region_df['e2e_latency_ms'].quantile(0.95)
        
        # Add statistics annotations
        plt.annotate(f'Median: {ttft_median:.1f}ms\n95th: {ttft_95th:.1f}ms', 
                    xy=(1, ttft_median), xytext=(0.6, ttft_median),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate(f'Median: {e2e_median:.1f}ms\n95th: {e2e_95th:.1f}ms', 
                    xy=(2, e2e_median), xytext=(2.4, e2e_median),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.title(f'{region_service} LLM Latency Distribution{title_suffix}')
        plt.xticks([1, 2], labels)
        plt.ylabel('Latency (ms)')
        plt.grid(True, alpha=0.3)
        
        # Ensure y-axis shows full range with some padding
        max_latency = max(
            region_df['e2e_latency_ms'].max(),
            region_df['ttft_ms'].max()
        )
        plt.ylim(0, max_latency * 1.2)  # 20% padding
        
        # Save the plot
        model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
        region_service_safe = region_service.replace(' ', '_').replace('-', '_').lower()
        output_file = output_dir / f'latency_distribution_{region_service_safe}{model_suffix}.png'
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved latency distribution plot for {region_service} to {output_file}")

def plot_region_comparison(df: pd.DataFrame, output_dir: Path, model_filter: Optional[str] = None):
    """Plot region comparison metrics."""
    if model_filter:
        df = df[df['model'] == model_filter]
        title_suffix = f" - {model_filter}"
    else:
        title_suffix = ""
    
    plt.figure(figsize=(14, 6))
    
    # Define vibrant colors for bars
    bar_colors = {
        'ttft': '#00A4EF',    # Bright blue
        'e2e': '#FF3366',     # Bright pink
        'tps': '#7FBA00'      # Bright green
    }
    
    # Get unique region/service combinations
    df['region_service'] = df['region'] + ' - ' + df['service']
    
    # Calculate mean latencies and throughput for each region
    region_metrics = df.groupby('region_service').agg({
        'ttft_ms': ['mean', 'std'],
        'e2e_latency_ms': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std']
    }).round(2)
    
    # Sort regions alphabetically
    region_services = sorted(region_metrics.index)
    region_metrics = region_metrics.reindex(region_services)
    
    # Prepare data for plotting
    x = np.arange(len(region_services))
    width = 0.25  # Narrower bars to fit three metrics
    
    # Create grouped bar plot
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot TTFT and E2E latency on left y-axis
    ttft_bars = ax1.bar(x - width, region_metrics['ttft_ms']['mean'], 
                       width, label='TTFT', color=bar_colors['ttft'])
    e2e_bars = ax1.bar(x, region_metrics['e2e_latency_ms']['mean'], 
                      width, label='E2E Latency', color=bar_colors['e2e'])
    
    # Add error bars
    ax1.errorbar(x - width, region_metrics['ttft_ms']['mean'],
               yerr=region_metrics['ttft_ms']['std'],
               fmt='none', color='black', capsize=5)
    ax1.errorbar(x, region_metrics['e2e_latency_ms']['mean'],
               yerr=region_metrics['e2e_latency_ms']['std'],
               fmt='none', color='black', capsize=5)
    
    # Create second y-axis for tokens per second
    ax2 = ax1.twinx()
    tps_bars = ax2.bar(x + width, region_metrics['tokens_per_second']['mean'], 
                      width, label='Tokens/Second', color=bar_colors['tps'])
    
    # Add error bars for tokens per second
    ax2.errorbar(x + width, region_metrics['tokens_per_second']['mean'],
               yerr=region_metrics['tokens_per_second']['std'],
               fmt='none', color='black', capsize=5)
    
    # Customize plot
    ax1.set_ylabel('Latency (ms)')
    ax2.set_ylabel('Tokens per Second')
    ax1.set_title(f'LLM Region Comparison{title_suffix}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(region_services, rotation=45, ha='right')
    
    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0,
                       fontsize=8)
    
    autolabel(ttft_bars, ax1)
    autolabel(e2e_bars, ax1)
    autolabel(tps_bars, ax2)
    
    plt.tight_layout()
    
    # Save the plot
    model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
    output_file = output_dir / f'region_comparison{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved region comparison plot to {output_file}")

def calculate_statistics(df: pd.DataFrame, model_filter: Optional[str] = None) -> pd.DataFrame:
    """Calculate comprehensive statistics for each region/service."""
    if model_filter:
        df = df[df['model'] == model_filter]
    
    # Get unique region/service combinations
    df['region_service'] = df['region'] + ' - ' + df['service']
    
    stats = []
    for region_service in df['region_service'].unique():
        region_df = df[df['region_service'] == region_service]
        
        # Get the model(s) for this region/service
        models = region_df['model'].unique()
        model_str = ', '.join(models)
        
        region_stats = {
            'Region-Service': region_service,
            'Model(s)': model_str,
            'Mean TTFT (ms)': region_df['ttft_ms'].mean(),
            'Median TTFT (ms)': region_df['ttft_ms'].median(),
            'Std TTFT (ms)': region_df['ttft_ms'].std(),
            '95th Percentile TTFT (ms)': region_df['ttft_ms'].quantile(0.95),
            '99th Percentile TTFT (ms)': region_df['ttft_ms'].quantile(0.99),
            'Mean E2E (ms)': region_df['e2e_latency_ms'].mean(),
            'Median E2E (ms)': region_df['e2e_latency_ms'].median(),
            'Std E2E (ms)': region_df['e2e_latency_ms'].std(),
            '95th Percentile E2E (ms)': region_df['e2e_latency_ms'].quantile(0.95),
            '99th Percentile E2E (ms)': region_df['e2e_latency_ms'].quantile(0.99),
            'Mean Tokens/Second': region_df['tokens_per_second'].mean(),
            'Median Tokens/Second': region_df['tokens_per_second'].median(),
            'Std Tokens/Second': region_df['tokens_per_second'].std(),
            'Mean Input Tokens': region_df['input_tokens'].mean() if 'input_tokens' in region_df.columns else None,
            'Mean Output Tokens': region_df['output_tokens'].mean() if 'output_tokens' in region_df.columns else None,
            'Generation Time (ms)': (region_df['e2e_latency_ms'] - region_df['ttft_ms']).mean(),
            'Sample Size': len(region_df)
        }
        stats.append(region_stats)
    
    # Round all numeric values to 2 decimal places
    stats_df = pd.DataFrame(stats)
    numeric_columns = stats_df.select_dtypes(include=['float64']).columns
    stats_df[numeric_columns] = stats_df[numeric_columns].round(2)
    
    return stats_df

def analyze_conversation_patterns(df: pd.DataFrame, output_dir: Path, model_filter: Optional[str] = None):
    """Analyze and visualize conversation patterns."""
    if model_filter:
        df = df[df['model'] == model_filter]
        title_suffix = f" - {model_filter}"
    else:
        title_suffix = ""
    
    # Plot conversation length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='conversation_length', hue='region', bins=30, alpha=0.6)
    plt.title(f'Conversation Length Distribution{title_suffix}')
    plt.xlabel('Conversation Length (characters)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
    output_file = output_dir / f'conversation_length_distribution{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot tokens per character ratio
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='region', y='tokens_per_char')
    plt.title(f'Tokens per Character Ratio by Region{title_suffix}')
    plt.xlabel('Region')
    plt.ylabel('Tokens per Character')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_file = output_dir / f'tokens_per_char_ratio{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

def analyze_response_patterns(responses_dir: Path, output_dir: Path, model_filter: Optional[str] = None):
    """Analyze patterns in the actual responses if available."""
    if not responses_dir or not responses_dir.exists():
        logger.warning("Responses directory not provided or does not exist. Skipping response pattern analysis.")
        return
    
    response_files = list(responses_dir.glob('*.json'))
    if not response_files:
        logger.warning("No response files found in the specified directory.")
        return
    
    # Collect response data
    response_data = []
    for file in response_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if model_filter and data.get('model') != model_filter:
                    continue
                response_data.append(data)
        except Exception as e:
            logger.warning(f"Error reading response file {file}: {str(e)}")
    
    if not response_data:
        logger.warning("No valid response data found for analysis.")
        return
    
    # Convert to DataFrame for analysis
    responses_df = pd.DataFrame(response_data)
    
    # Analyze completion patterns
    plt.figure(figsize=(12, 6))
    completion_lengths = responses_df['completion'].str.len()
    sns.histplot(data=completion_lengths, bins=30)
    plt.title('Response Length Distribution')
    plt.xlabel('Response Length (characters)')
    plt.ylabel('Count')
    plt.tight_layout()
    
    model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
    output_file = output_dir / f'response_length_distribution{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

def plot_latency_vs_conversation_length(df: pd.DataFrame, output_dir: Path, model_filter: Optional[str] = None):
    """Plot latency metrics against conversation length."""
    if model_filter:
        df = df[df['model'] == model_filter]
        title_suffix = f" - {model_filter}"
    else:
        title_suffix = ""
    
    # Create scatter plots with trend lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # TTFT vs Conversation Length
    sns.regplot(data=df, x='conversation_length', y='ttft_ms', ax=ax1, scatter_kws={'alpha':0.5})
    ax1.set_title(f'TTFT vs Conversation Length{title_suffix}')
    ax1.set_xlabel('Conversation Length (characters)')
    ax1.set_ylabel('TTFT (ms)')
    
    # E2E Latency vs Conversation Length
    sns.regplot(data=df, x='conversation_length', y='e2e_latency_ms', ax=ax2, scatter_kws={'alpha':0.5})
    ax2.set_title(f'E2E Latency vs Conversation Length{title_suffix}')
    ax2.set_xlabel('Conversation Length (characters)')
    ax2.set_ylabel('E2E Latency (ms)')
    
    plt.tight_layout()
    
    model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
    output_file = output_dir / f'latency_vs_length{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

def calculate_conversation_statistics(df: pd.DataFrame, model_filter: Optional[str] = None) -> pd.DataFrame:
    """Calculate statistics specific to conversation-based testing."""
    if model_filter:
        df = df[df['model'] == model_filter]
    
    # Get unique region/service combinations
    df['region_service'] = df['region'] + ' - ' + df['service']
    
    stats = []
    for region_service in df['region_service'].unique():
        region_df = df[df['region_service'] == region_service]
        
        # Get the model(s) for this region/service
        models = region_df['model'].unique()
        model_str = ', '.join(models)
        
        region_stats = {
            'Region-Service': region_service,
            'Model(s)': model_str,
            'Mean Conversation Length': region_df['conversation_length'].mean(),
            'Median Conversation Length': region_df['conversation_length'].median(),
            'Mean Input Tokens': region_df['input_tokens'].mean(),
            'Mean Output Tokens': region_df['output_tokens'].mean(),
            'Mean Tokens/Char': region_df['tokens_per_char'].mean(),
            'Mean TTFT/Char (ms/char)': region_df['ttft_ms'] / region_df['conversation_length'].mean(),
            'Mean E2E/Char (ms/char)': region_df['e2e_latency_ms'] / region_df['conversation_length'].mean(),
            'Sample Size': len(region_df)
        }
        stats.append(region_stats)
    
    # Round all numeric values to 2 decimal places
    stats_df = pd.DataFrame(stats)
    numeric_columns = stats_df.select_dtypes(include=['float64']).columns
    stats_df[numeric_columns] = stats_df[numeric_columns].round(2)
    
    return stats_df

def get_latest_metrics_file(data_dir: Path) -> Path:
    """Get the most recent metrics CSV file from the specified directory."""
    metrics_files = list(data_dir.glob('llm_metrics_*.csv'))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found in {data_dir}")
    return max(metrics_files, key=lambda p: p.stat().st_mtime)

def plot_latency_boxplot_comparison(df: pd.DataFrame, output_dir: Path, model_filter: Optional[str] = None):
    """Create comprehensive box-whisker plots comparing latency metrics across regions."""
    if model_filter:
        df = df[df['model'] == model_filter]
        title_suffix = f" - {model_filter}"
    else:
        title_suffix = ""
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Color palette for regions
    region_colors = {
        'eastus': '#FF3366',      # Pink
        'westus': '#00A4EF',      # Blue
        'westeurope': '#7FBA00',  # Green
        'southcentralus': '#FFB900', # Yellow
        'northeurope': '#737373',  # Gray
        'sweden': '#8661C5',      # Purple
        'india': '#FF8C00',       # Orange
    }
    
    # TTFT Box Plot
    sns.boxplot(data=df, x='region', y='ttft_ms', ax=ax1, 
                palette=[region_colors.get(r, '#8661C5') for r in df['region'].unique()])
    ax1.set_title(f'TTFT Distribution by Region{title_suffix}')
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Time to First Token (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add numeric annotations for median, min, max
    for i, region in enumerate(df['region'].unique()):
        region_data = df[df['region'] == region]['ttft_ms']
        median = region_data.median()
        minimum = region_data.min()
        maximum = region_data.max()
        ax1.text(i, median, f'Median: {median:.1f}', horizontalalignment='center', verticalalignment='bottom')
        ax1.text(i, minimum, f'Min: {minimum:.1f}', horizontalalignment='center', verticalalignment='top')
        ax1.text(i, maximum, f'Max: {maximum:.1f}', horizontalalignment='center', verticalalignment='bottom')
    
    # E2E Latency Box Plot
    sns.boxplot(data=df, x='region', y='e2e_latency_ms', ax=ax2,
                palette=[region_colors.get(r, '#8661C5') for r in df['region'].unique()])
    ax2.set_title(f'E2E Latency Distribution by Region{title_suffix}')
    ax2.set_xlabel('Region')
    ax2.set_ylabel('End-to-End Latency (ms)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add numeric annotations for median, min, max
    for i, region in enumerate(df['region'].unique()):
        region_data = df[df['region'] == region]['e2e_latency_ms']
        median = region_data.median()
        minimum = region_data.min()
        maximum = region_data.max()
        ax2.text(i, median, f'Median: {median:.1f}', horizontalalignment='center', verticalalignment='bottom')
        ax2.text(i, minimum, f'Min: {minimum:.1f}', horizontalalignment='center', verticalalignment='top')
        ax2.text(i, maximum, f'Max: {maximum:.1f}', horizontalalignment='center', verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
    output_file = output_dir / f'latency_boxplot_comparison{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved latency boxplot comparison to {output_file}")

def plot_latency_vs_tokens(df: pd.DataFrame, output_dir: Path, model_filter: Optional[str] = None):
    """Create scatter plots showing latency vs input token length with trend lines for each region."""
    if model_filter:
        df = df[df['model'] == model_filter]
        title_suffix = f" - {model_filter}"
    else:
        title_suffix = ""
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Color palette for regions
    region_colors = {
        'eastus': '#FF3366',      # Pink
        'westus': '#00A4EF',      # Blue
        'westeurope': '#7FBA00',  # Green
        'southcentralus': '#FFB900', # Yellow
        'northeurope': '#737373',  # Gray
        'sweden': '#8661C5',      # Purple
        'india': '#FF8C00',       # Orange
    }
    
    # Plot TTFT vs Input Tokens
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        color = region_colors.get(region, '#8661C5')
        
        # Scatter plot with trend line for TTFT
        sns.regplot(data=region_data, x='input_tokens', y='ttft_ms',
                   scatter_kws={'alpha':0.5, 'color': color},
                   line_kws={'color': color},
                   label=region, ax=ax1)
        
        # Calculate and add correlation coefficient
        correlation = region_data['input_tokens'].corr(region_data['ttft_ms'])
        ax1.text(0.05, 0.95 - df['region'].unique().tolist().index(region) * 0.05,
                f'{region}: r = {correlation:.2f}',
                transform=ax1.transAxes, color=color)
    
    ax1.set_title(f'TTFT vs Input Tokens{title_suffix}')
    ax1.set_xlabel('Input Tokens')
    ax1.set_ylabel('Time to First Token (ms)')
    ax1.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot E2E Latency vs Input Tokens
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        color = region_colors.get(region, '#8661C5')
        
        # Scatter plot with trend line for E2E latency
        sns.regplot(data=region_data, x='input_tokens', y='e2e_latency_ms',
                   scatter_kws={'alpha':0.5, 'color': color},
                   line_kws={'color': color},
                   label=region, ax=ax2)
        
        # Calculate and add correlation coefficient
        correlation = region_data['input_tokens'].corr(region_data['e2e_latency_ms'])
        ax2.text(0.05, 0.95 - df['region'].unique().tolist().index(region) * 0.05,
                f'{region}: r = {correlation:.2f}',
                transform=ax2.transAxes, color=color)
    
    ax2.set_title(f'E2E Latency vs Input Tokens{title_suffix}')
    ax2.set_xlabel('Input Tokens')
    ax2.set_ylabel('End-to-End Latency (ms)')
    ax2.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    model_suffix = f"_{model_filter.replace('-', '_')}" if model_filter else ""
    output_file = output_dir / f'latency_vs_tokens{model_suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved latency vs tokens plot to {output_file}")

def main():
    """Main entry point for the analysis script."""
    args = parse_args()
    
    # Create output directory with timestamp
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Analysis results will be saved to: {output_dir}")
    
    # Get input file
    if args.input:
        input_file = Path(args.input)
    else:
        data_dir = Path('llm_metrics_analysis_output')
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        input_file = get_latest_metrics_file(data_dir)
    
    logger.info(f"Using metrics file: {input_file}")
    
    # Load and process data
    df = load_metrics_data(input_file)
    
    # Generate all visualizations and statistics
    plot_latency_over_time(df, output_dir, args.model)
    plot_tokens_per_second(df, output_dir, args.model)
    plot_latency_distribution(df, output_dir, args.model)
    plot_region_comparison(df, output_dir, args.model)
    plot_latency_boxplot_comparison(df, output_dir, args.model)
    plot_latency_vs_tokens(df, output_dir, args.model)
    plot_latency_vs_conversation_length(df, output_dir, args.model)
    
    # Calculate and save statistics
    stats_df = calculate_statistics(df, args.model)
    stats_file = output_dir / 'latency_statistics.csv'
    stats_df.to_csv(stats_file)
    logger.info(f"Saved statistics to {stats_file}")
    
    # Analyze conversation patterns
    analyze_conversation_patterns(df, output_dir, args.model)
    
    # Analyze response patterns if responses directory is provided
    if args.responses_dir:
        responses_dir = Path(args.responses_dir)
        if responses_dir.exists():
            analyze_response_patterns(responses_dir, output_dir, args.model)
        else:
            logger.warning(f"Responses directory not found: {responses_dir}")
    
    # Calculate and save conversation statistics
    conv_stats_df = calculate_conversation_statistics(df, args.model)
    conv_stats_file = output_dir / 'conversation_statistics.csv'
    conv_stats_df.to_csv(conv_stats_file)
    logger.info(f"Saved conversation statistics to {conv_stats_file}")
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    main() 