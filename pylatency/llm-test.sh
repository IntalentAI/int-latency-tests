#!/bin/bash

# Set timestamp for output directory
timestamp=$(date +"%Y%m%d_%H%M%S")
base_output_dir="llm_metrics_analysis_output/${timestamp}"

# Create output directory
mkdir -p "$base_output_dir"

# Log file
log_file="${base_output_dir}/test_run.log"
exec 1> >(tee -a "$log_file") 2>&1

echo "Starting LLM Test Runs at $(date)"
echo "Output directory: ${base_output_dir}"

# Function to run tests
run_test() {
    local test_name=$1
    local stream=$2
    local mode=$3
    local output_dir="${base_output_dir}/${test_name}"
    
    echo "=== Starting Test: ${test_name} ==="
    echo "Stream: ${stream}"
    echo "Mode: ${mode}"
    
    # Create output directory
    mkdir -p "${output_dir}"
    
    # Run Groq and Azure tests together
    echo "Running combined Groq and Azure tests..."
    uv run python run_llm_load.py \
        --services groq azure \
        --groq-models specdec versatile \
        --regions eastus sweden india \
        --mode "${mode}" \
        $([ "${stream}" = "true" ] && echo "--stream") \
        --iterations 10 \
        --save-responses
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Error: Test failed with exit code ${exit_code}"
        exit $exit_code
    fi
    
    echo "=== Completed Test: ${test_name} ==="
    echo
}

# Run 1: Tools with Streaming
echo "Starting Run 1: Tools with Streaming"
run_test "tools_streaming" "true" "tools"

# Run 2: Chat without Streaming
echo "Starting Run 2: Chat without Streaming"
run_test "chat_no_streaming" "false" "chat"

# Generate analysis for each run
echo "Generating analysis reports..."
for run in "tools_streaming" "chat_no_streaming"; do
    echo "Analyzing ${run}..."
    if [ -f "${base_output_dir}/${run}/llm_metrics.csv" ]; then
        uv run python llm_metrics_analysis.py \
            --input "${base_output_dir}/${run}/llm_metrics.csv" \
            --output "${base_output_dir}/${run}/analysis"
    else
        echo "Warning: Metrics file not found for ${run}"
    fi
done

echo "All tests completed at $(date)"
echo "Results and analysis available in: ${base_output_dir}"

# Print summary of test configurations
cat << EOF > "${base_output_dir}/test_summary.txt"
LLM Test Summary
===============

Test Configurations:
1. Tools with Streaming
   - Services: Groq and Azure (running in parallel)
   - Groq Models: specdec, versatile
   - Azure Regions: eastus, sweden, india
   - Streaming: enabled
   - Mode: tools
   - Iterations: 10

2. Chat without Streaming
   - Services: Groq and Azure (running in parallel)
   - Groq Models: specdec, versatile
   - Azure Regions: eastus, sweden, india
   - Streaming: disabled
   - Mode: chat
   - Iterations: 10

Output Structure:
${base_output_dir}/
├── tools_streaming/
│   ├── llm_metrics.csv
│   ├── responses/
│   └── analysis/
├── chat_no_streaming/
│   ├── llm_metrics.csv
│   ├── responses/
│   └── analysis/
├── test_run.log
└── test_summary.txt

Generated: $(date)
EOF

echo "Test summary written to: ${base_output_dir}/test_summary.txt" 