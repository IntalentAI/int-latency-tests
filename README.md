#  Latency Benchmarks 

A tool for measuring and comparing speech (and potentially LLM's) latency across major cloud providers (Azure, Deepgram, OpenAI).
This will help us understand the latency of each service and how it affects the user experience from various regions. 

## Features

- ✅ Multi-service latency testing (Azure Cognitive Services, Deepgram, OpenAI TTS)
- ⏱️ Measures Time-to-First-Byte (TTFB) and end-to-end latency
- 📊 Automated metrics collection and CSV export
- 📈 Built-in visualization of latency metrics
- 🔊 Supports real-time audio streaming analysis

## Setup

1. **Install dependencies** (using Python 3.13+):
```bash
uv venv
uv pip install -r pyproject.toml
```

2. Configure environment - copy template and add API keys:
```bash
cp env-template .env
```

3. Configuration
Add these to your .env file:
```
AZURE_SPEECH_KEY=<your-key>
AZURE_REGION=<region>
DEEPGRAM_API_KEY=<your-key>
OPENAI_API_KEY=<your-key>
AZURE_STORAGE_CONN_STRING=<optional>
```
3. Run TTS Latency Suite:
```bash
uv run python run_tts_load.py \
  --iterations 200 \
  --wait-time 30 \
  --cleanup
```
Arguments:

```
--iterations: Number of test runs per service (default: 60)
--wait-time: Cool-down between requests in seconds (default: 10)
--cleanup: Remove generated audio files after test (recommended)
--save-audio: Save audio files to disk (default: True)
--no-save-audio: Do not save audio files to disk (default: False)
```
4. Analyze Metrics
```bash
uv run python tts_metrics_analysis.py
```

Output:
``` 
data/
├── metrics_analysis_output (plots, stats etc.)
├── <.wav files generated if not cleaned up>
└── <metrics files in csv format>
```


Metrics Collected:
Time-to-First-Byte (TTFB)
End-to-End latency
Service availability

Supported Services:
Azure Cognitive Services
Deepgram
OpenAI TTS