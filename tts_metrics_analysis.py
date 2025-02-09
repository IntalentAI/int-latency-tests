def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze TTS metrics and generate visualizations')
    parser.add_argument('--input', type=str, required=True,
                      help='Input directory containing metrics CSV files')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory for analysis results')
    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get latest metrics file
    try:
        metrics_file = get_latest_metrics_file(input_dir)
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
        raise

if __name__ == "__main__":
    import argparse
    main() 