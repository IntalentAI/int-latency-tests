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
    def __init__(self, service_name: str, region: str):
        self.service_name = service_name
        self.region = region
        self.metrics: List[SpeechMetricEntry] = []
    
    def add_metric(self, entry: SpeechMetricEntry):
        self.metrics.append(entry)
    
    def save_to_csv(self, filename: str):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Service', 'Region', 'Input Text', 'TTFB (ms)', 
                           'E2E Latency (ms)', 'Start Time', 'End Time'])
            for metric in self.metrics:
                writer.writerow([
                    self.service_name,
                    self.region,
                    metric.input_text,
                    metric.ttfb_ms,
                    metric.e2e_latency_ms,
                    metric.start_time,
                    metric.end_time
                ])

class AzureSpeechSynthesizer:
    def __init__(
        self,
        api_key: str,
        region: str,
        voice: str = AZURE_DEFAULT_VOICE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        params: Optional[Dict[str, Any]] = None
    ):
        self.api_key = api_key
        self.region = region
        self.voice_id = voice
        self.sample_rate = sample_rate
        self.metrics = SpeechMetrics("Azure", region)
        
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
        
        # Set up event handlers
        self.speech_synthesizer.synthesizing.connect(self._handle_synthesizing)
        self.speech_synthesizer.synthesis_completed.connect(self._handle_completed)
        self.speech_synthesizer.synthesis_canceled.connect(self._handle_canceled)
        logger.info("Azure speech synthesizer initialized")

    def _handle_synthesizing(self, evt):
        """Handle audio chunks as they arrive."""
        if evt.result and evt.result.audio_data:
            logger.debug(f"Received audio chunk: {len(evt.result.audio_data)} bytes")
            try:
                self.audio_queue.put_nowait(evt.result.audio_data)
            except asyncio.QueueFull:
                logger.warning("Audio queue is full, dropping chunk")

    def _handle_completed(self, evt):
        """Handle synthesis completion."""
        logger.info("Speech synthesis completed")
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
            logger.info(f"Starting synthesis for text: {text}")

            # Start synthesis with SSML
            ssml = self._construct_ssml(text)
            logger.debug(f"Generated SSML: {ssml}")
            
            # Create synthesis result
            result = await asyncio.to_thread(
                lambda: self.speech_synthesizer.speak_ssml_async(ssml).get()
            )
            
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.CancellationDetails(result)
                logger.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                logger.error(f"Error details: {cancellation_details.error_details}")
                raise Exception(f"Speech synthesis canceled: {cancellation_details.error_details}")

            # Stream audio chunks as they arrive
            timeout = 30  # 30 seconds timeout
            while True:
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout)
                    if chunk is None:  # End of stream
                        logger.info("Received end of stream signal")
                        break

                    if not ttfb_recorded:
                        ttfb_time = (time.time() - start_time) * 1000
                        logger.info(f"First chunk received. TTFB: {ttfb_time:.2f}ms")
                        self.metrics.add_metric(SpeechMetricEntry(
                            input_text=text,
                            ttfb_ms=ttfb_time,
                            e2e_latency_ms=None,
                            start_time=start_time,
                            end_time=None
                        ))
                        ttfb_recorded = True

                    yield AudioFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate
                    )
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for audio chunks")
                    break

            # Update final metrics
            end_time = time.time()
            if len(self.metrics.metrics) > 0:
                last_metric = self.metrics.metrics[-1]
                last_metric.e2e_latency_ms = (end_time - start_time) * 1000
                last_metric.end_time = end_time
                logger.info(f"Synthesis completed. E2E latency: {last_metric.e2e_latency_ms:.2f}ms")

        except Exception as e:
            logger.error(f"Azure synthesis failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def synthesize_speech(self, text: str, output_file: str) -> bool:
        """Generate speech and save to file."""
        try:
            logger.info(f"Starting speech synthesis to file: {output_file}")
            audio_data = bytearray()
            
            async for frame in self.synthesize_speech_stream(text):
                logger.info(f"Received frame: {len(frame.audio)} bytes")
                audio_data.extend(frame.audio)
            
            with wave.open(output_file, 'wb') as wav_file:
                logger.info(f"Writing audio data to {output_file}")
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
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
        self.metrics = SpeechMetrics("Deepgram", "global")

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
                    self.metrics.add_metric(SpeechMetricEntry(
                        input_text=text,
                        ttfb_ms=ttfb_time,
                        e2e_latency_ms=None,
                        start_time=start_time,
                        end_time=None
                    ))
                    ttfb_recorded = True
                    first_chunk = False

                yield AudioFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate
                )

            # Update final metrics
            end_time = time.time()
            if len(self.metrics.metrics) > 0:
                last_metric = self.metrics.metrics[-1]
                last_metric.e2e_latency_ms = (end_time - start_time) * 1000
                last_metric.end_time = end_time

        except Exception as e:
            print(f"Deepgram synthesis failed: {str(e)}")
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
            print(f"Failed to save audio: {str(e)}")
            return False

class OpenAISpeechSynthesizer:
    """OpenAI Text-to-Speech service implementation."""
    
    def __init__(
        self,
        api_key: str,
        voice: str = OPENAI_DEFAULT_VOICE,
        model: Literal["tts-1", "tts-1-hd"] = "tts-1-hd",
        sample_rate: int = OPENAI_SAMPLE_RATE,  # Changed default to use OpenAI's native rate
    ):
        self.api_key = api_key
        self.voice_id = voice
        self.model = model
        self.sample_rate = OPENAI_SAMPLE_RATE  # Always use OpenAI's native rate
        self.client = AsyncOpenAI(api_key=api_key)
        self.metrics = SpeechMetrics("OpenAI", "global")

    async def synthesize_speech_stream(self, text: str) -> AsyncGenerator[AudioFrame, None]:
        """Generate speech audio stream from text."""
        try:
            start_time = time.time()
            ttfb_recorded = False
            logger.info(f"Starting OpenAI synthesis for text: {text}")

            response = await self.client.audio.speech.create(
                input=text or " ",  # Text must contain at least one character
                model=self.model,
                voice=self.voice_id,
                response_format="pcm",
            )
            
            # Get response content as bytes
            audio_data = response.content
            
            if not ttfb_recorded:
                ttfb_time = (time.time() - start_time) * 1000
                logger.info(f"First chunk received. TTFB: {ttfb_time:.2f}ms")
                self.metrics.add_metric(SpeechMetricEntry(
                    input_text=text,
                    ttfb_ms=ttfb_time,
                    e2e_latency_ms=None,
                    start_time=start_time,
                    end_time=None
                ))
                ttfb_recorded = True

            # Yield audio data in chunks
            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield AudioFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate
                )

            # Update final metrics
            end_time = time.time()
            if len(self.metrics.metrics) > 0:
                last_metric = self.metrics.metrics[-1]
                last_metric.e2e_latency_ms = (end_time - start_time) * 1000
                last_metric.end_time = end_time
                logger.info(f"Synthesis completed. E2E latency: {last_metric.e2e_latency_ms:.2f}ms")

        except Exception as e:
            logger.error(f"OpenAI synthesis failed: {str(e)}", exc_info=True)
            raise

    async def synthesize_speech(self, text: str, output_file: str) -> bool:
        """Generate speech and save to file."""
        try:
            logger.info(f"Starting OpenAI speech synthesis to file: {output_file}")
            audio_data = bytearray()
            chunks_received = 0
            
            async for frame in self.synthesize_speech_stream(text):
                chunks_received += 1
                logger.debug(f"Received frame {chunks_received}: {len(frame.audio)} bytes")
                audio_data.extend(frame.audio)
            
            if not audio_data:
                logger.error("No audio data received from synthesis stream")
                return False
                
            logger.info(f"Writing {len(audio_data)} bytes to WAV file")
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            logger.info(f"Successfully saved OpenAI audio to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save OpenAI audio: {str(e)}", exc_info=True)
            logger.exception("Full traceback:")
            return False

def upload_to_azure_storage(local_zip_path: str, connection_string: str, container_name: str):
    """Upload a file to Azure Blob Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
        
        blob_name = os.path.basename(local_zip_path)
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(local_zip_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"Successfully uploaded {blob_name} to Azure Storage")
        return True
    except Exception as e:
        print(f"Failed to upload to Azure Storage: {str(e)}")
        return False

async def run_speech_test(
    azure_key: str,
    azure_region: str,
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
    azure_synthesizer = AzureSpeechSynthesizer(
        azure_key, 
        azure_region,
        sample_rate=AUDIO_SAMPLE_RATE
    )
    deepgram_synthesizer = DeepgramSpeechSynthesizer(
        deepgram_key,
        sample_rate=AUDIO_SAMPLE_RATE
    )
    openai_synthesizer = OpenAISpeechSynthesizer(
        openai_key,
        sample_rate=AUDIO_SAMPLE_RATE
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting speech test at {timestamp}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    openai_audio_data = None  # Initialize variable
    deepgram_audio_data = None  # Initialize variable
    azure_audio_data = None  # Initialize variable

    metrics_file = os.path.join(data_dir, f"speech_metrics_{timestamp}.csv")
    
    # Write CSV header
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Service', 'Region', 'Input Text', 'TTFB (ms)', 
                        'E2E Latency (ms)', 'Start Time', 'End Time'])

    def flush_metrics_to_disk():
        """Write current metrics to CSV file"""
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write Azure metrics
            for metric in azure_synthesizer.metrics.metrics:
                writer.writerow([
                    azure_synthesizer.metrics.service_name,
                    azure_synthesizer.metrics.region,
                    metric.input_text,
                    metric.ttfb_ms,
                    metric.e2e_latency_ms,
                    metric.start_time,
                    metric.end_time
                ])
            # Write Deepgram metrics
            for metric in deepgram_synthesizer.metrics.metrics:
                writer.writerow([
                    deepgram_synthesizer.metrics.service_name,
                    deepgram_synthesizer.metrics.region,
                    metric.input_text,
                    metric.ttfb_ms,
                    metric.e2e_latency_ms,
                    metric.start_time,
                    metric.end_time
                ])
            # Write OpenAI metrics
            for metric in openai_synthesizer.metrics.metrics:
                writer.writerow([
                    openai_synthesizer.metrics.service_name,
                    openai_synthesizer.metrics.region,
                    metric.input_text,
                    metric.ttfb_ms,
                    metric.e2e_latency_ms,
                    metric.start_time,
                    metric.end_time
                ])
        # Clear metrics after writing
        azure_synthesizer.metrics.metrics.clear()
        deepgram_synthesizer.metrics.metrics.clear()
        openai_synthesizer.metrics.metrics.clear()
        logger.info("Flushed metrics to disk")

    def save_audio_batch(batch_num):
        """Save current audio data to files"""
        if not save_audio:
            logger.info("Skipping audio save as save_audio=False")
            return []
            
        logger.info(f"Saving audio batch {batch_num}")
        output_files = []
        if openai_audio_data is not None:
            openai_output = os.path.join(data_dir, f"openai_speech_{timestamp}_batch{batch_num}.wav")
            with wave.open(openai_output, 'wb') as wav_file:
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wav_file.setframerate(OPENAI_SAMPLE_RATE)
                wav_file.writeframes(openai_audio_data.tobytes())
            output_files.append(openai_output)
            logger.info(f"OpenAI audio batch {batch_num} saved")

        if deepgram_audio_data is not None:
            deepgram_output = os.path.join(data_dir, f"deepgram_speech_{timestamp}_batch{batch_num}.wav")
            with wave.open(deepgram_output, 'wb') as wav_file:
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wav_file.setframerate(AUDIO_SAMPLE_RATE)
                wav_file.writeframes(deepgram_audio_data.tobytes())
            output_files.append(deepgram_output)
            logger.info(f"Deepgram audio batch {batch_num} saved")

        if azure_audio_data is not None:
            azure_output = os.path.join(data_dir, f"azure_speech_{timestamp}_batch{batch_num}.wav")
            with wave.open(azure_output, 'wb') as wav_file:
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wav_file.setframerate(AUDIO_SAMPLE_RATE)
                wav_file.writeframes(azure_audio_data.tobytes())
            output_files.append(azure_output)
            logger.info(f"Azure audio batch {batch_num} saved")
        return output_files

    all_output_files = []
    batch_size = 5
    for i in range(num_iterations):
        logger.info(f"Generating Sentence for Iteration {i+1}/{num_iterations}")
        sentence = generator.generate_sentence()
        print(f"\nIteration {i+1}/{num_iterations}")
        print(f"Text: {sentence}")
        
        if save_audio:
            # OpenAI synthesis
            openai_temp = os.path.join(data_dir, f"openai_temp_{i}.wav")
            logger.info(f"Synthesizing OpenAI for {sentence}")
            openai_success = await openai_synthesizer.synthesize_speech(sentence, openai_temp)
            if openai_success:
                with wave.open(openai_temp, 'rb') as wav_file:
                    current_audio = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
                    silence = np.zeros(24000, dtype=np.int16)
                    openai_audio_data = current_audio if openai_audio_data is None else np.concatenate([openai_audio_data, silence, current_audio])
                os.remove(openai_temp)
                logger.info(f"OpenAI synthesis completed for {sentence}")
            
            # Deepgram synthesis
            deepgram_temp = os.path.join(data_dir, f"deepgram_temp_{i}.wav")
            logger.info(f"Synthesizing deepgram for {sentence}")
            deepgram_success = await deepgram_synthesizer.synthesize_speech(sentence, deepgram_temp)
            if deepgram_success:
                with wave.open(deepgram_temp, 'rb') as wav_file:
                    current_audio = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
                    silence = np.zeros(24000, dtype=np.int16)
                    deepgram_audio_data = current_audio if deepgram_audio_data is None else np.concatenate([deepgram_audio_data, silence, current_audio])
                os.remove(deepgram_temp)
                logger.info(f"Deepgram synthesis completed for {sentence}")
            
            # Azure synthesis
            azure_temp = os.path.join(data_dir, f"azure_temp_{i}.wav")
            logger.info(f"Synthesizing azure for {sentence}")
            azure_success = await azure_synthesizer.synthesize_speech(sentence, azure_temp)
            if azure_success:
                with wave.open(azure_temp, 'rb') as wav_file:
                    current_audio = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
                    silence = np.zeros(24000, dtype=np.int16)
                    azure_audio_data = current_audio if azure_audio_data is None else np.concatenate([azure_audio_data, silence, current_audio])
                os.remove(azure_temp)
            logger.info(f"Azure synthesis completed for {sentence}")
        else:
            # Just synthesize without saving concatenated audio
            openai_success = await openai_synthesizer.synthesize_speech(sentence, os.devnull)
            deepgram_success = await deepgram_synthesizer.synthesize_speech(sentence, os.devnull)
            azure_success = await azure_synthesizer.synthesize_speech(sentence, os.devnull)
        
        # Flush data every 5 iterations
        if (i + 1) % batch_size == 0 or i == num_iterations - 1:
            batch_num = (i + 1) // batch_size
            logger.info(f"Saving batch {batch_num}")
            flush_metrics_to_disk()
            batch_files = save_audio_batch(batch_num)
            all_output_files.extend(batch_files)
            # Reset audio data
            openai_audio_data = None
            deepgram_audio_data = None
            azure_audio_data = None

        if i < num_iterations - 1:
            logger.info(f"Waiting for {wait_time} seconds before next iteration")
            time.sleep(wait_time)
    
    # Create final zip file with all batches
    zip_filename = os.path.join(data_dir, f"speech_data_{timestamp}.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for output_file in all_output_files:
            zipf.write(output_file, os.path.basename(output_file))
        zipf.write(metrics_file, os.path.basename(metrics_file))
    logger.info(f"Created zip file: {zip_filename}")
    
    if os.getenv("UPLOAD_TO_AZURE_STORAGE") == "true":
        # Upload and cleanup
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER")
        
        if connection_string and container_name:
            if upload_to_azure_storage(zip_filename, connection_string, container_name):
                if cleanup:
                    # Clean up local files after successful upload
                    os.remove(zip_filename)
                    for output_file in all_output_files:
                        os.remove(output_file)
                    os.remove(metrics_file)
                    print("Local files cleaned up after upload")
        else:
            print("Azure Storage credentials not found in environment variables")
            print(f"Files are saved locally in {data_dir}")

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
    print("Loading environment variables")
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    
    # Get environment variables
    azure_speech_key = os.getenv('AZURE_SPEECH_API_KEY')
    azure_speech_region = os.getenv('AZURE_SPEECH_REGION')
    deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # Validate required variables
    if not all([azure_speech_key, azure_speech_region, deepgram_api_key, openai_api_key]):
        raise ValueError("Required API credentials not found in .env file")
    
    logger.info(f"Starting speech test with {args.iterations} iterations and {args.wait_time}s wait time")
    
    # Run the async test
    asyncio.run(run_speech_test(
        azure_speech_key,
        azure_speech_region,
        deepgram_api_key,
        openai_api_key,
        num_iterations=args.iterations,
        wait_time=args.wait_time,
        cleanup=False,
        save_audio=args.save_audio
    ))