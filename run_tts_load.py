# Run Latency Benchmarks for TTS

from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
import time
from dataclasses import dataclass
from typing import List, Optional, AsyncGenerator, Dict, Any, Literal
import os
import csv
import numpy as np
import wave
import zipfile
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    SpeakOptions,
)
import asyncio
from loguru import logger
from openai import AsyncOpenAI
import argparse
import sys
import random
import tempfile
import shutil
import httpx
import aiofiles
import subprocess

# Audio configuration constants
AUDIO_SAMPLE_RATE = 16000  # Standard sample rate for all services
AUDIO_CHANNELS = 1         # Mono audio
AUDIO_SAMPLE_WIDTH = 2     # 16-bit audio
OPENAI_SAMPLE_RATE = 24000  # OpenAI TTS native sample rate

# Service-specific constants
AZURE_DEFAULT_VOICE = "en-US-JennyNeural"
DEEPGRAM_DEFAULT_VOICE = "aura-helios-en"
OPENAI_DEFAULT_VOICE = "alloy"

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
logger.add(
    f"log_latency_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    level="DEBUG",
)

class FileBasedSentenceGenerator:
    def __init__(self, file_path="data/input/questions.txt"):
        """Initialize with path to questions file."""
        self.file_path = file_path
        self.questions = []
        
        # Load questions from file
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.questions = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.questions)} questions from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load questions from {file_path}: {str(e)}")
            raise

    def generate_sentence(self, min_words: int = 8, max_words: int = 16) -> str:
        """Return a random question from the file. min_words and max_words are ignored."""
        if not self.questions:
            raise ValueError("No questions loaded from file")
            
        # Get a random question
        question = random.choice(self.questions)
        
        logger.info(f"Selected question: {question}")
        return question
 
@dataclass
class SpeechMetricEntry:
    service_name: str
    region: str
    input_text: str
    ttfb_ms: float
    e2e_latency_ms: float
    start_time: float
    end_time: float

@dataclass
class AudioFrame:
    audio: bytes
    sample_rate: int
    num_channels: int = 1

class SpeechMetrics:
    def __init__(self):
        self.metrics: List[SpeechMetricEntry] = []
    
    def add_metric(self, service_name: str, region: str, input_text: str, ttfb_ms: float, 
                  e2e_latency_ms: Optional[float] = None, start_time: float = None, 
                  end_time: Optional[float] = None):
        self.metrics.append(SpeechMetricEntry(
            service_name=service_name,
            region=region,
            input_text=input_text,
            ttfb_ms=ttfb_ms,
            e2e_latency_ms=e2e_latency_ms,
            start_time=start_time,
            end_time=end_time
        ))
    
    def save_to_csv(self, filename: str):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Service', 'Region', 'Input Text', 'TTFB (ms)', 
                           'E2E Latency (ms)', 'Start Time', 'End Time'])
            for metric in self.metrics:
                writer.writerow([
                    metric.service_name,
                    metric.region,
                    metric.input_text,
                    metric.ttfb_ms,
                    metric.e2e_latency_ms,
                    metric.start_time,
                    metric.end_time
                ])

# Create global metrics collector
speech_metrics = SpeechMetrics()

def flush_metrics_to_disk():
    """Save current metrics to CSV file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('data'):
        os.makedirs('data')
    speech_metrics.save_to_csv(f'data/run_metrics_{timestamp}.csv')

class AzureSpeechSynthesizer:
    def __init__(
        self,
        api_key: str,
        region: str,
        endpoint: str = None,
        voice: str = AZURE_DEFAULT_VOICE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        params: Optional[Dict[str, Any]] = None
    ):
        self.api_key = api_key
        self.region = region
        self.voice_id = voice
        self.sample_rate = sample_rate
        self.endpoint = endpoint
        
        # Default settings
        self.settings = {
            "language": "en-US",
            "rate": "1.05",
            "pitch": None,
            "volume": None,
            "style": None,
            "style_degree": None,
            "role": None,
            "emphasis": None
        }
        if params:
            self.settings.update(params)
        
        self.speech_config = None
        self.speech_synthesizer = None
        self.audio_queue = asyncio.Queue()
        
        logger.info(f"Initialized Azure TTS with voice: {voice}, region: {region}")

    def _construct_ssml(self, text: str) -> str:
        """Construct SSML with voice and prosody settings."""
        language = self.settings["language"]
        ssml = (
            f"<speak version='1.0' xml:lang='{language}' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{self.voice_id}'>"
            "<mstts:silence type='Sentenceboundary' value='20ms' />"
        )

        if self.settings["style"]:
            ssml += f"<mstts:express-as style='{self.settings['style']}'"
            if self.settings["style_degree"]:
                ssml += f" styledegree='{self.settings['style_degree']}'"
            if self.settings["role"]:
                ssml += f" role='{self.settings['role']}'"
            ssml += ">"

        prosody_attrs = []
        if self.settings["rate"]:
            prosody_attrs.append(f"rate='{self.settings['rate']}'")
        if self.settings["pitch"]:
            prosody_attrs.append(f"pitch='{self.settings['pitch']}'")
        if self.settings["volume"]:
            prosody_attrs.append(f"volume='{self.settings['volume']}'")

        ssml += f"<prosody {' '.join(prosody_attrs)}>"

        if self.settings["emphasis"]:
            ssml += f"<emphasis level='{self.settings['emphasis']}'>"

        ssml += text

        if self.settings["emphasis"]:
            ssml += "</emphasis>"

        ssml += "</prosody>"

        if self.settings["style"]:
            ssml += "</mstts:express-as>"

        ssml += "</voice></speak>"
        return ssml

    async def initialize(self):
        """Initialize speech config and synthesizer."""
        logger.info("Initializing Azure speech synthesizer")
        if self.endpoint:
            self.speech_config = speechsdk.SpeechConfig(
                endpoint=self.endpoint,
                subscription=self.api_key,
                speech_recognition_language=self.settings["language"]
            )
        else:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.api_key,
                region=self.region,
                speech_recognition_language=self.settings["language"]
            )
        
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )
        
        # Set service property for websocket connection
        self.speech_config.set_service_property(
            "synthesizer.synthesis.connection.synthesisConnectionImpl",
            "websocket",
            speechsdk.ServicePropertyChannel.UriQueryParameter
        )
        
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None
        )

        # Establish and open the connection
        self.connection = speechsdk.Connection.from_speech_synthesizer(self.speech_synthesizer)
        self.connection.open(True)
        logger.info("Connection opened successfully")
        
        # Set up event handlers
        self.speech_synthesizer.synthesizing.connect(self._handle_synthesizing)
        self.speech_synthesizer.synthesis_completed.connect(self._handle_completed)
        self.speech_synthesizer.synthesis_canceled.connect(self._handle_canceled)
        logger.info("Azure speech synthesizer initialized")

    def _handle_synthesizing(self, evt):
        """Handle audio chunks as they arrive."""
        if evt.result and evt.result.audio_data:
            logger.trace(f"Received audio chunk: {len(evt.result.audio_data)} bytes")
            try:
                self.audio_queue.put_nowait(evt.result.audio_data)
            except asyncio.QueueFull:
                logger.warning("Audio queue is full, dropping chunk")

    def _handle_completed(self, evt):
        """Handle synthesis completion."""
        logger.debug("Speech synthesis completed")
        try:
            self.audio_queue.put_nowait(None)  # Signal completion
        except asyncio.QueueFull:
            logger.warning("Could not send completion signal, queue is full")

    def _handle_canceled(self, evt):
        """Handle synthesis cancellation."""
        error_details = evt.error_details if hasattr(evt, 'error_details') else "Unknown error"
        logger.error(f"Speech synthesis canceled: {error_details}")
        try:
            self.audio_queue.put_nowait(None)  # Signal completion
        except asyncio.QueueFull:
            logger.warning("Could not send cancellation signal, queue is full")

    async def synthesize_speech_stream(self, text: str) -> AsyncGenerator[AudioFrame, None]:
        """Generate speech audio stream from text."""
        try:
            if not self.speech_synthesizer:
                await self.initialize()

            start_time = time.time()
            ttfb_recorded = False
            
            # Use a bounded queue to prevent memory issues
            self.audio_queue = asyncio.Queue(maxsize=100)
            
            # Create synthesis result - run in threadpool to avoid blocking
            result_future = asyncio.create_task(
                asyncio.to_thread(
                    self.speech_synthesizer.speak_ssml_async(self._construct_ssml(text)).get
                )
            )

            # Create a task for queue monitoring
            queue_monitor = asyncio.create_task(self._monitor_queue())
            
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                        
                        if chunk is None:  # End of stream
                            break

                        if not ttfb_recorded:
                            ttfb_time = (time.time() - start_time) * 1000
                            speech_metrics.add_metric(
                                service_name="Azure",
                                region=self.region,
                                input_text=text,
                                ttfb_ms=ttfb_time,
                                start_time=start_time
                            )
                            ttfb_recorded = True

                        yield AudioFrame(audio=chunk, sample_rate=self.sample_rate)
                        self.audio_queue.task_done()

                    except asyncio.TimeoutError:
                        if result_future.done():
                            # Synthesis is complete but no more data
                            break
                        # Otherwise continue waiting
                        continue

            finally:
                # Clean up tasks
                queue_monitor.cancel()
                try:
                    await queue_monitor
                except asyncio.CancelledError:
                    pass
                
                if not result_future.done():
                    result_future.cancel()
                    try:
                        await result_future
                    except asyncio.CancelledError:
                        pass

            # Update final metrics
            end_time = time.time()
            for metric in speech_metrics.metrics:
                if (metric.service_name == "Azure" and 
                    metric.input_text == text and 
                    metric.e2e_latency_ms is None):
                    metric.e2e_latency_ms = (end_time - start_time) * 1000
                    metric.end_time = end_time
                    break

        except Exception as e:
            logger.error(f"Azure synthesis failed: {str(e)}", exc_info=True)
            raise

    async def _monitor_queue(self):
        """Monitor queue health and log issues."""
        while True:
            await asyncio.sleep(0.1)  # Check every 100ms
            qsize = self.audio_queue.qsize()
            if qsize > 80:  # 80% full
                logger.warning(f"Audio queue is {qsize}% full")

    async def synthesize_speech(self, text: str, output_file: str, save_audio: bool = True) -> bool:
        """Generate speech and save to file."""
        try:
            logger.info(f"Starting speech synthesis to file: {output_file}")
            audio_data = bytearray()
            
            # Create a separate task for stream processing
            async def process_stream():
                async for frame in self.synthesize_speech_stream(text):
                    audio_data.extend(frame.audio)
            
            # Run stream processing with increased timeout
            try:
                await asyncio.wait_for(process_stream(), timeout=15.0)  # Increased from 5.0 to 15.0
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing audio stream for text: {text[:50]}...")
                return False
            except Exception as e:
                logger.error(f"Error processing audio stream: {str(e)}")
                return False
            
            if not audio_data:
                logger.error("No audio data received from synthesis stream")
                return False
            
            # Write WAV file in a thread pool to avoid blocking
            if save_audio:
                def write_wav():
                    with wave.open(output_file, 'wb') as wav_file:
                        wav_file.setnchannels(AUDIO_CHANNELS)
                        wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                        wav_file.setframerate(self.sample_rate)
                        wav_file.writeframes(audio_data)
                
                await asyncio.to_thread(write_wav)
                logger.info(f"Successfully saved audio to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}", exc_info=True)
            return False


class DeepgramSpeechSynthesizer:
    def __init__(
        self,
        api_key: str,
        voice: str = DEEPGRAM_DEFAULT_VOICE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        encoding: str = "linear16"
    ):
        self.settings = {
            "encoding": encoding,
        }
        self.voice = voice
        self.sample_rate = sample_rate
        self.client = DeepgramClient(api_key=api_key)

    async def synthesize_speech_stream(self, text: str) -> AsyncGenerator[AudioFrame, None]:
        try:
            start_time = time.time()
            ttfb_recorded = False

            # Configure TTS options
            options = SpeakOptions(
                model=self.voice,
                encoding=self.settings["encoding"],
                sample_rate=self.sample_rate,
                container="none",
            )

            # Get streaming response
            response = await asyncio.to_thread(
                self.client.speak.v("1").stream,
                {"text": text},
                options
            )

            # Get audio buffer
            audio_buffer = response.stream_memory
            if audio_buffer is None:
                raise ValueError("No audio data received from Deepgram")

            # Read and yield the audio data in chunks
            audio_buffer.seek(0)
            chunk_size = 8192

            # Record TTFB on first chunk
            first_chunk = True
            while True:
                chunk = audio_buffer.read(chunk_size)
                if not chunk:
                    break

                if first_chunk and not ttfb_recorded:
                    ttfb_time = (time.time() - start_time) * 1000
                    speech_metrics.add_metric(
                        service_name="Deepgram",
                        region="global",
                        input_text=text,
                        ttfb_ms=ttfb_time,
                        start_time=start_time
                    )
                    ttfb_recorded = True
                    first_chunk = False

                yield AudioFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate
                )

            # Update final metrics
            end_time = time.time()
            for metric in speech_metrics.metrics:
                if (metric.service_name == "Deepgram" and 
                    metric.input_text == text and 
                    metric.e2e_latency_ms is None):
                    metric.e2e_latency_ms = (end_time - start_time) * 1000
                    metric.end_time = end_time
                    break

        except Exception as e:
            logger.error(f"Deepgram synthesis failed: {str(e)}")
            raise

    async def synthesize_speech(self, text: str, output_file: str) -> bool:
        try:
            audio_data = bytearray()
            
            # Collect all audio frames
            async for frame in self.synthesize_speech_stream(text):
                audio_data.extend(frame.audio)
            
            # Write WAV file
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}")
            return False

class OpenAISpeechSynthesizer:
    """OpenAI Text-to-Speech service implementation."""
    
    def __init__(
        self,
        api_key: str,
        voice: str = OPENAI_DEFAULT_VOICE,
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",  # Using faster model
        sample_rate: int = OPENAI_SAMPLE_RATE,
    ):
        self.api_key = api_key
        self.voice_id = voice
        self.model = model
        self.sample_rate = OPENAI_SAMPLE_RATE
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=10.0,
            max_retries=2,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=2.0,    # Connection timeout
                    read=8.0,       # Read timeout
                    write=2.0,      # Write timeout
                    pool=None       # No pool timeout needed
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10
                ),
                trust_env=False
            )
        )

    async def synthesize_speech(self, text: str, output_file: str, save_audio: bool = True) -> bool:
        """Generate speech and save to file using streaming."""
        try:
            logger.info(f"Starting OpenAI streaming synthesis: {text[:50]}...")
            start_time = time.time()
            audio_buffer = bytearray()
            ttfb_recorded = False
            
            # Use streaming response with PCM format
            async with self.client.audio.speech.with_streaming_response.create(
                input=text,
                model=self.model,
                voice=self.voice_id,
                response_format="pcm",
                speed=1.1,  # Slightly faster synthesis
            ) as response:
                # Process chunks as they arrive
                async for chunk in response.iter_bytes(chunk_size=4096):  # Smaller chunks for faster TTFB
                    if not ttfb_recorded:
                        ttfb_time = (time.time() - start_time) * 1000
                        logger.info(f"OpenAI streaming first chunk. TTFB: {ttfb_time:.2f}ms")
                        speech_metrics.add_metric(
                            service_name="OpenAI",
                            region="global",
                            input_text=text,
                            ttfb_ms=ttfb_time,
                            start_time=start_time
                        )
                        ttfb_recorded = True
                    
                    audio_buffer.extend(chunk)
                    # Could add real-time processing here if needed

            # Record streaming completion time before file operations
            streaming_end_time = time.time()
            streaming_latency = (streaming_end_time - start_time) * 1000
            logger.info(f"OpenAI streaming completed. Latency: {streaming_latency:.2f}ms")

            if save_audio:
                with wave.open(output_file, 'wb') as wav_file:
                    wav_file.setnchannels(AUDIO_CHANNELS)
                    wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_buffer)

            # Update metrics
            for metric in speech_metrics.metrics:
                if (metric.service_name == "OpenAI" and 
                    metric.input_text == text and 
                    metric.e2e_latency_ms is None):
                    metric.e2e_latency_ms = streaming_latency
                    metric.end_time = streaming_end_time
                    break

            logger.info(f"Successfully saved OpenAI audio to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save OpenAI audio: {str(e)}", exc_info=True)
            return False

    async def synthesize_speech_stream(self, text: str) -> AsyncGenerator[AudioFrame, None]:
        """Generate speech audio stream from text."""
        try:
            start_time = time.time()
            ttfb_recorded = False
            
            # Create speech request with timeout
            response = await asyncio.wait_for(
                self.client.audio.speech.create(
                    input=text,
                    model=self.model,
                    voice=self.voice_id,
                    response_format="pcm",
                ),
                timeout=5.0
            )
            
            # Get response content as bytes
            audio_data = response.content
            
            if not ttfb_recorded:
                ttfb_time = (time.time() - start_time) * 1000
                logger.debug(f"OpenAI first chunk received. TTFB: {ttfb_time:.2f}ms")
                speech_metrics.add_metric(
                    service_name="OpenAI",
                    region="global",
                    input_text=text,
                    ttfb_ms=ttfb_time,
                    start_time=start_time
                )
                ttfb_recorded = True

            # Yield audio data in chunks using a thread pool
            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield AudioFrame(audio=chunk, sample_rate=self.sample_rate)

            # Update final metrics
            end_time = time.time()
            e2e_latency = (end_time - start_time) * 1000
            logger.info(f"OpenAI synthesis completed. E2E latency: {e2e_latency:.2f}ms")
            
            for metric in speech_metrics.metrics:
                if (metric.service_name == "OpenAI" and 
                    metric.input_text == text and 
                    metric.e2e_latency_ms is None):
                    metric.e2e_latency_ms = e2e_latency
                    metric.end_time = end_time
                    break

        except asyncio.TimeoutError:
            logger.error("OpenAI synthesis timed out")
            raise
        except Exception as e:
            logger.error(f"OpenAI synthesis failed: {str(e)}", exc_info=True)
            raise

def upload_to_azure_storage(local_zip_path: str, connection_string: str, container_name: str):
    """Upload a file to Azure Blob Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
        
        blob_name = f"{os.uname().nodename}_{os.path.basename(local_zip_path)}"
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(local_zip_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info(f"Successfully uploaded {blob_name} to Azure Storage")
        return True
    except Exception as e:
        logger.info(f"Failed to upload to Azure Storage: {str(e)}")
        return False

async def run_speech_test(
    azure_key: str,
    azure_region: str,
    azure_endpoint: str,
    deepgram_key: str,
    openai_key: str,
    num_iterations: int = 100,
    wait_time: int = 10,
    cleanup: bool = True,
    save_audio: bool = True
):
    logger.info(f"Starting speech test with {num_iterations} iterations")
    generator = FileBasedSentenceGenerator()
    logger.info("FileBasedSentenceGenerator created")
    
    # Initialize services based on environment variables
    run_azure = os.getenv('RUN_AZURE_TTS', 'false').lower() == 'true'
    run_deepgram = os.getenv('RUN_DEEPGRAM_TTS', 'false').lower() == 'true'
    run_openai = os.getenv('RUN_OPENAI_TTS', 'false').lower() == 'true'
    
    logger.info(f"Services enabled - Azure: {run_azure}, Deepgram: {run_deepgram}, OpenAI: {run_openai}")
    
    # Initialize only required services
    azure_synthesizer = None
    deepgram_synthesizer = None
    openai_synthesizer = None
    
    if run_azure:
        azure_synthesizer = AzureSpeechSynthesizer(
            azure_key, 
            azure_region,
            azure_endpoint,
            sample_rate=AUDIO_SAMPLE_RATE
        )
        
    if run_deepgram:
        deepgram_synthesizer = DeepgramSpeechSynthesizer(
            deepgram_key,
            sample_rate=AUDIO_SAMPLE_RATE
        )
        
    if run_openai:
        openai_synthesizer = OpenAISpeechSynthesizer(
            openai_key,
            sample_rate=AUDIO_SAMPLE_RATE
        )

    # Create output directory if saving audio
    output_dir = None
    if save_audio:
        output_dir = os.path.join('data', 'output', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving audio files to {output_dir}")

    # Main test loop
    for i in range(num_iterations):
        logger.info(f"Iteration {i+1}/{num_iterations}")
        sentence = generator.generate_sentence()
        logger.info(f"Text: {sentence}")
        
        # Create temporary files for this iteration
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as azure_temp, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as deepgram_temp, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as openai_temp:
            
            azure_success = False
            deepgram_success = False
            openai_success = False
            
            # Only run enabled services
            if run_azure:
                logger.info(f"Synthesizing Azure for {sentence}")
                azure_success = await azure_synthesizer.synthesize_speech(sentence, azure_temp.name, save_audio)
                
            if run_deepgram:
                logger.info(f"Synthesizing Deepgram for {sentence}")
                deepgram_success = await deepgram_synthesizer.synthesize_speech(sentence, deepgram_temp.name)
                
            if run_openai:
                logger.info(f"Synthesizing OpenAI for {sentence}")
                openai_success = await openai_synthesizer.synthesize_speech(sentence, openai_temp.name, save_audio)
            
            # Copy files to output directory if save_audio is True
            if save_audio and output_dir:
                timestamp = datetime.now().strftime('%H%M%S')
                if run_azure and azure_success:
                    output_file = os.path.join(output_dir, f'azure_{i+1}_{timestamp}.wav')
                    await asyncio.to_thread(shutil.copy2, azure_temp.name, output_file)
                    logger.info(f"Saved Azure audio to {output_file}")
                
                if run_deepgram and deepgram_success:
                    output_file = os.path.join(output_dir, f'deepgram_{i+1}_{timestamp}.wav')
                    await asyncio.to_thread(shutil.copy2, deepgram_temp.name, output_file)
                    logger.info(f"Saved Deepgram audio to {output_file}")
                
                if run_openai and openai_success:
                    output_file = os.path.join(output_dir, f'openai_{i+1}_{timestamp}.wav')
                    await asyncio.to_thread(shutil.copy2, openai_temp.name, output_file)
                    logger.info(f"Saved OpenAI audio to {output_file}")

            # Clean up temporary files if requested
            if cleanup:
                for temp_file in [azure_temp.name, deepgram_temp.name, openai_temp.name]:
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass
        
        # Wait between iterations if specified
        if wait_time > 0 and i < num_iterations - 1:
            await asyncio.sleep(wait_time)
        
        # Flush metrics every 10 iterations
        if (i + 1) % 10 == 0:
            flush_metrics_to_disk()
    
    # Final metrics flush
    flush_metrics_to_disk()
    logger.info("Speech test completed")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TTS comparison test')
    parser.add_argument('--iterations', type=int, default=60,
                      help='Number of iterations to run (default: 60)')
    parser.add_argument('--wait-time', type=int, default=1,
                      help='Wait time between iterations in seconds (default: 1)')
    parser.add_argument('--save-audio', action='store_true', default=True,
                      help='Save audio files to disk (default: True)')
    parser.add_argument('--no-save-audio', action='store_false', dest='save_audio',
                      help='Do not save audio files to disk')
    args = parser.parse_args()

    # Load environment variables from .env file
    logger.info("Loading environment variables")
    logger.info("Loading environment variables")
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    
    # Get environment variables
    azure_speech_key = os.getenv('AZURE_SPEECH_API_KEY')
    azure_speech_region = os.getenv('AZURE_SPEECH_REGION')
    azure_endpoint = os.getenv('AZURE_ENDPOINT')
    deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # Only validate required variables based on enabled services
    run_azure = os.getenv('RUN_AZURE_TTS', 'false').lower() == 'true'
    run_deepgram = os.getenv('RUN_DEEPGRAM_TTS', 'false').lower() == 'true'
    run_openai = os.getenv('RUN_OPENAI_TTS', 'false').lower() == 'true'
    
    if run_azure and not all([azure_speech_key, azure_speech_region]):
        raise ValueError("Azure Speech credentials not found in .env file")
    if run_deepgram and not deepgram_api_key:
        raise ValueError("Deepgram API key not found in .env file")
    if run_openai and not openai_api_key:
        raise ValueError("OpenAI API key not found in .env file")
    
    if not any([run_azure, run_deepgram, run_openai]):
        raise ValueError("No TTS services enabled. Set at least one of RUN_AZURE_TTS, RUN_DEEPGRAM_TTS, or RUN_OPENAI_TTS to true")
    
    logger.info(f"Starting speech test with {args.iterations} iterations and {args.wait_time}s wait time")
    
    # Run the async test
    asyncio.run(run_speech_test(
        azure_speech_key,
        azure_speech_region,
        azure_endpoint,
        deepgram_api_key,
        openai_api_key,
        num_iterations=args.iterations,
        wait_time=args.wait_time,
        cleanup=False,
        save_audio=args.save_audio
    ))