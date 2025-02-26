# Run Latency Benchmarks for LLMs

from datetime import datetime
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal, Union
import os
import csv
import numpy as np
import argparse
import sys
import random
import asyncio
import httpx
from loguru import logger
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
import json
from pathlib import Path

# Load environment variables
load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Create base output directory
base_output_dir = "llm_metrics_analysis_output"
os.makedirs(base_output_dir, exist_ok=True)

# Configure file logging with timestamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger.add(
    f"{base_output_dir}/log_llm_latency_{timestamp}.log",
    level="DEBUG",
)

# Interview questions pool
INTERVIEW_QUESTIONS = [
    "Tell me about your experience with distributed systems.",
    "How do you approach testing in your development process?",
    "What's your experience with microservices architecture?",
    "How do you handle technical debt in your projects?",
    "Describe a challenging performance optimization you've implemented.",
    "How do you approach system design problems?",
    "What's your experience with CI/CD pipelines?",
    "How do you handle production incidents?",
    "Tell me about a time you had to make a difficult technical decision.",
    "How do you stay current with technology trends?"
]

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_next_question",
            "description": "Get a random next interview question from the pool",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "repeat_question",
            "description": "Repeat the current question",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

@dataclass
class LLMMetricEntry:
    """Data class for storing LLM latency metrics."""
    service_name: str
    region: str
    model: str
    input_text: str
    ttft_ms: float  # Time to first token
    e2e_latency_ms: float  # End-to-end latency
    tokens_per_second: float  # Throughput
    input_tokens: int  # Number of input tokens
    output_tokens: int  # Number of output tokens
    start_time: float  # Unix timestamp
    end_time: float  # Unix timestamp
    tool_calls: bool  # Whether tool calls were used
    tool_call_count: int  # Number of tool calls made
    tool_call_latency_ms: float  # Total time spent in tool calls

class LLMMetrics:
    """Class for collecting and saving LLM metrics."""
    def __init__(self):
        self.metrics: List[LLMMetricEntry] = []
    
    def add_metric(self, service_name: str, region: str, model: str, input_text: str, 
                  ttft_ms: float, e2e_latency_ms: float, tokens_per_second: float,
                  input_tokens: int, output_tokens: int, start_time: float, end_time: float,
                  tool_calls: bool = False, tool_call_count: int = 0, tool_call_latency_ms: float = 0.0):
        self.metrics.append(LLMMetricEntry(
            service_name=service_name,
            region=region,
            model=model,
            input_text=input_text,
            ttft_ms=ttft_ms,
            e2e_latency_ms=e2e_latency_ms,
            tokens_per_second=tokens_per_second,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            start_time=start_time,
            end_time=end_time,
            tool_calls=tool_calls,
            tool_call_count=tool_call_count,
            tool_call_latency_ms=tool_call_latency_ms
        ))
    
    def save_to_csv(self, filename: str):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Service', 'Region', 'Model', 'Input Text', 'TTFT (ms)', 
                           'E2E Latency (ms)', 'Tokens/Second', 'Input Tokens',
                           'Output Tokens', 'Start Time', 'End Time', 'Tool Calls Used',
                           'Tool Call Count', 'Tool Call Latency (ms)'])
            for metric in self.metrics:
                writer.writerow([
                    metric.service_name,
                    metric.region,
                    metric.model,
                    metric.input_text[:50] + "..." if len(metric.input_text) > 50 else metric.input_text,
                    metric.ttft_ms,
                    metric.e2e_latency_ms,
                    metric.tokens_per_second,
                    metric.input_tokens,
                    metric.output_tokens,
                    metric.start_time,
                    metric.end_time,
                    metric.tool_calls,
                    metric.tool_call_count,
                    metric.tool_call_latency_ms
                ])

# Create global metrics collector
llm_metrics = LLMMetrics()

def flush_metrics_to_disk():
    """Save current metrics to CSV file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, 'llm_metrics.csv')
    llm_metrics.save_to_csv(metrics_file)
    logger.info(f"Saved metrics to {metrics_file}")
    return metrics_file

# Tool implementation functions
def get_next_question() -> str:
    """Get a random next interview question."""
    return random.choice(INTERVIEW_QUESTIONS)

def repeat_question(current_question: str) -> str:
    """Repeat the current question."""
    return current_question

# System prompt for interview assistant
INTERVIEW_SYSTEM_PROMPT = """
You are an AI interview assistant that conducts interview conversations between an interviewer and a candidate.
Your task is to determine the next action the interviewer agent should take based on the conversation history.

You have access to the following tools:
1. get_next_question: Get a random next interview question when the current question is sufficiently answered
2. repeat_question: Repeat the current question when the answer is unclear or incomplete

Analyze the conversation carefully to identify the next action the interviewer agent should take, focusing on:
- Whether the candidate directly answered the question asked
- The completeness and clarity of the candidate's response
- Whether the candidate provided concrete examples when appropriate
- The depth and thoughtfulness of the candidate's answer

Based on your analysis:
1. If the answer is complete and satisfactory, use get_next_question to move forward
2. If the answer is unclear or incomplete, use repeat_question to ask again
3. If you need clarification, provide a follow-up question in your response

"""

class InterviewConversationGenerator:
    """Generates interview conversations for LLM testing."""
    def __init__(self, file_path="data/input/interview_conversations.txt"):
        """Initialize with path to conversations file or use default conversations."""
        self.file_path = file_path
        self.conversations = []
        
        # Try to load conversations from file
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by a marker that separates conversations
                raw_conversations = content.split('===CONVERSATION===')
                for conv in raw_conversations:
                    if conv.strip():
                        self.conversations.append(conv.strip())
            logger.info(f"Loaded {len(self.conversations)} conversations from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load conversations from {file_path}: {str(e)}")
            # Create default conversations if file not found
            self.conversations = [
                """Hi, This is an interview for a software engineer position. Please answer the questions to the best of your ability. Are you ready to start?""",
                """Interviewer: Tell me about your experience with Python programming.
Candidate: I've been using Python for about 5 years now. I started with data analysis using pandas and numpy, then moved into web development with Django and Flask.
Interviewer: Can you give a specific example of a challenging Python project you worked on?
Candidate: Sure, I built a real-time analytics dashboard using Django, Celery, and Redis that processed data from multiple sources. The challenge was handling the high volume of incoming data while keeping the dashboard responsive.""",

                """Interviewer: How do you handle conflicts in a team setting?
Candidate: I believe communication is key. I try to understand everyone's perspective before suggesting solutions.
Interviewer: Could you share a specific situation where you resolved a team conflict?
Candidate: In my previous role, two team members disagreed on the architecture for a new feature. I organized a whiteboarding session where each person could explain their approach, and we ended up creating a hybrid solution that incorporated the best elements of both.""",

                """Interviewer: Describe your experience with cloud technologies.
Candidate: I've worked extensively with AWS services like EC2, S3, and Lambda. I've also used Azure for some projects.
Interviewer: How have you implemented security best practices in your cloud deployments?
Candidate: I always follow the principle of least privilege for IAM roles and use security groups to restrict access. I also implement encryption for data at rest and in transit.""",

                """Interviewer: What's your approach to debugging complex issues?
Candidate: I start by reproducing the issue and gathering as much information as possible. Then I isolate components to identify where the problem is occurring.
Interviewer: Can you walk me through a particularly difficult bug you solved?
Candidate: I once debugged a memory leak in a Node.js application that only occurred in production. By adding incremental logging and analyzing memory snapshots, I discovered that we were caching large objects without proper cleanup.""",

                """Interviewer: How do you stay updated with new technologies in your field?
Candidate: I follow several tech blogs and participate in online communities. I also try to build small projects with new technologies to get hands-on experience.""",

                """Interviewer: Tell me about a time when you had to meet a tight deadline.
Candidate: In my last job, we had a major client deliverable due in two weeks, but our team lead unexpectedly went on medical leave. I stepped up to coordinate the team, prioritized features, and we delivered on time.
Interviewer: What specific strategies did you use to ensure the deadline was met?
Candidate: I created a detailed task breakdown with daily milestones, held brief daily stand-ups to address blockers, and negotiated with stakeholders to reduce some non-critical requirements.""",

                """Interviewer: How do you approach learning a new programming language or framework?
Candidate: I usually start with the official documentation to understand the core concepts, then build a small project to apply what I've learned. I also look for patterns and similarities with technologies I already know.""",

                """Interviewer: Describe your experience with database design and optimization.
Candidate: I've designed schemas for both relational and NoSQL databases. I'm experienced with normalization, indexing strategies, and query optimization.
Interviewer: Can you give an example of how you optimized database performance in a previous role?
Candidate: I noticed our application was slowing down due to inefficient queries. I added appropriate indexes, rewrote some queries to use joins more effectively, and implemented query caching, which reduced average query time by 70%.""",

                """Interviewer: How do you handle receiving critical feedback?
Candidate: I view feedback as an opportunity to improve. I try not to take it personally, focus on understanding the specific issues, and create an action plan to address them.""",

                """Interviewer: What experience do you have with agile development methodologies?
Candidate: I've worked in Scrum teams for the past 4 years. I'm familiar with sprint planning, daily stand-ups, retrospectives, and continuous integration practices.
Interviewer: How do you ensure quality while maintaining the pace of development in an agile environment?
Candidate: I believe in building quality in from the start through test-driven development, automated testing, and code reviews. I also advocate for technical debt management as part of our regular sprint work."""
            ]
            logger.info(f"Using {len(self.conversations)} default conversations")

    def generate_conversation(self) -> str:
        """Return a random conversation from the collection."""
        if not self.conversations:
            raise ValueError("No conversations available")
            
        # Get a random conversation
        conversation = random.choice(self.conversations)
        
        logger.info(f"Selected conversation: {conversation[:50]}..." if len(conversation) > 50 else conversation)
        return conversation

class GroqCloudClient:
    """Client for GroqCloud API."""
    def __init__(
        self,
        api_key: str,
        model: str = None,
        region: str = "default",
        max_tokens: int = 8192,
        temperature: float = 0.5,
        timeout: float = 60.0,  # Increased default timeout
    ):
        self.api_key = api_key
        self.model = model or os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chunk_timing = []
        
        # Initialize the client with appropriate timeout settings
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout,
            max_retries=3,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0,    # Increased connection timeout
                    read=timeout,    # Use the full timeout for read operations
                    write=15.0,      # Increased write timeout
                    pool=None
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=10,  # Increased from 5
                    max_connections=20             # Increased from 10
                ),
                trust_env=False
            )
        )
        
        logger.info(f"Initialized GroqCloud client for {self.model} in {region}")

    def _parse_conversation_to_messages(self, conversation: str) -> List[Dict[str, str]]:
        """Parse a conversation string into a list of messages for the API."""
        lines = conversation.strip().split('\n')
        messages = [{"role": "system", "content": INTERVIEW_SYSTEM_PROMPT}]
        
        current_speaker = None
        current_message = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Interviewer:"):
                if current_speaker:
                    messages.append({"role": current_speaker, "content": current_message.strip()})
                current_speaker = "user"
                current_message = line.replace("Interviewer:", "").strip()
            elif line.startswith("Candidate:"):
                if current_speaker:
                    messages.append({"role": current_speaker, "content": current_message.strip()})
                current_speaker = "user"
                current_message = f"[Candidate's response]: {line.replace('Candidate:', '').strip()}"
            else:
                current_message += " " + line
        
        if current_speaker and current_message:
            messages.append({"role": current_speaker, "content": current_message.strip()})
        
        return messages

    async def generate_completion(self, conversation: str, use_tools: bool = True, stream: bool = False) -> Dict[str, Any]:
        """Generate a completion for the given conversation and measure latency metrics."""
        try:
            # Initial setup
            start_time = time.time()
            connection_start_time = time.time()  # Initialize with start_time
            first_chunk_time = None
            ttft_time = None
            first_token_time = None
            tool_call_count = 0
            tool_call_latency = 0.0
            full_response = ""
            tool_calls = []
            chunk_count = 0
            valid_chunk_count = 0
            
            # Parse messages before starting latency measurement
            messages = self._parse_conversation_to_messages(conversation)
            
            try:
                if stream:
                    # Streaming mode
                    response_stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stream=True,
                        tools=TOOLS if use_tools else None
                    )
                    
                    # Record first chunk arrival time
                    first_chunk_time = time.time()
                    
                    async for chunk in response_stream:
                        chunk_time = time.time()
                        chunk_count += 1
                        
                        if not chunk or not chunk.choices:
                            continue
                        
                        valid_chunk_count += 1
                        self.chunk_timing.append(chunk_time - start_time)
                        
                        # Record time to first token
                        if not first_token_time and chunk.choices and (
                            (hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content) or 
                            (hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls)
                        ):
                            first_token_time = chunk_time
                            ttft_time = (first_token_time - start_time) * 1000
                        
                        # Handle tool calls
                        if chunk.choices and hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                if hasattr(tool_call, 'function'):
                                    tool_call_data = {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                    tool_calls.append(tool_call_data)
                                    tool_call_count += 1
                        
                        # Accumulate response
                        if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                    
                    # If we never got a first token, use the last chunk time
                    if not first_token_time and chunk_count > 0:
                        first_token_time = time.time()
                        ttft_time = (first_token_time - start_time) * 1000
                
                else:
                    # Non-streaming mode
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stream=False,
                        tools=TOOLS if use_tools else None
                    )
                    
                    # Record TTFT (in non-streaming mode, this is the same as total latency)
                    first_token_time = time.time()
                    ttft_time = (first_token_time - start_time) * 1000
                    
                    # Process response
                    if response.choices:
                        # Handle tool calls
                        if hasattr(response.choices[0], 'tool_calls') and response.choices[0].tool_calls:
                            for tool_call in response.choices[0].tool_calls:
                                tool_call_data = {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.arguments
                                }
                                tool_calls.append(tool_call_data)
                                tool_call_count += 1
                        
                        # Get content
                        if hasattr(response.choices[0], 'message') and response.choices[0].message.content:
                            full_response = response.choices[0].message.content
            
            except Exception as api_error:
                logger.error(f"[GroqCloud] API error: {str(api_error)}")
                if hasattr(api_error, 'response'):
                    logger.error(f"Response status: {api_error.response.status_code}")
                    logger.error(f"Response body: {await api_error.response.text()}")
                raise
            
            # Process tool calls if any
            if tool_calls:
                tool_start_time = time.time()
                for tool_call in tool_calls:
                    if tool_call["name"] == "get_next_question":
                        next_question = get_next_question()
                        full_response += f"\nNext question: {next_question}"
                    elif tool_call["name"] == "repeat_question":
                        current_question = conversation.split("Interviewer:")[-1].split("Candidate:")[0].strip()
                        full_response += f"\nRepeating question: {current_question}"
                tool_call_latency = (time.time() - tool_start_time) * 1000
            
            end_time = time.time()
            e2e_latency = (end_time - start_time) * 1000
            
            # Calculate metrics
            try:
                input_tokens = len(conversation.split()) // 3 * 4
                output_tokens = len(full_response.split()) // 3 * 4
            except Exception as e:
                logger.warning(f"Could not get token counts: {str(e)}")
                input_tokens = len(conversation) // 4
                output_tokens = len(full_response) // 4
            
            tokens_per_second = 0
            if output_tokens > 0 and first_token_time:
                generation_time = (end_time - first_token_time)
                if generation_time > 0:
                    tokens_per_second = output_tokens / generation_time
            
            # Log everything after latency calculations are complete
            logger.info(f"[GroqCloud] Completion Summary:")
            logger.info(f"[GroqCloud] Input conversation: {conversation}")
            logger.info(f"[GroqCloud] Messages sent:\n{json.dumps(messages, indent=2)}")
            logger.info(f"[GroqCloud] Mode: {'tools' if use_tools else 'chat'}, Streaming: {stream}")
            if stream:
                logger.info(f"[GroqCloud] Received {chunk_count} chunks")
            if tool_calls:
                logger.info(f"[GroqCloud] Tool calls made ({len(tool_calls)}):\n{json.dumps(tool_calls, indent=2)}")
            logger.info(f"[GroqCloud] Final response:\n{full_response}")
            logger.info(f"[GroqCloud] Performance metrics:")
            logger.info(f"  - TTFT: {ttft_time:.2f}ms")
            logger.info(f"  - E2E Latency: {e2e_latency:.2f}ms")
            logger.info(f"  - Tokens/second: {tokens_per_second:.2f}")
            logger.info(f"  - Tool call latency: {tool_call_latency:.2f}ms")
            
            # Log detailed timing information
            logger.info(f"[GroqCloud] Detailed timing for {self.model}:")
            
            # Safely calculate and log timing metrics
            if connection_start_time and start_time:
                logger.info(f"  - Start to connection: {(connection_start_time - start_time) * 1000:.2f}ms")
            
            if first_chunk_time and connection_start_time:
                logger.info(f"  - Connection to first chunk: {(first_chunk_time - connection_start_time) * 1000:.2f}ms")
            
            if first_token_time and first_chunk_time:
                logger.info(f"  - First chunk to first token: {(first_token_time - first_chunk_time) * 1000:.2f}ms")
            elif stream:
                logger.info("  - No first token received")
            
            logger.info(f"  - Total chunks: {chunk_count} (Valid: {valid_chunk_count})")
            
            if self.chunk_timing:
                chunk_intervals = np.diff(self.chunk_timing) * 1000  # Convert to ms
                if len(chunk_intervals) > 0:
                    logger.info(f"  - Average chunk interval: {np.mean(chunk_intervals):.2f}ms")
                    logger.info(f"  - Max chunk interval: {np.max(chunk_intervals):.2f}ms")
            
            # Clear chunk timing for next run
            self.chunk_timing = []
            
            # Add metrics to collector
            llm_metrics.add_metric(
                service_name="GroqCloud",
                region=self.region,
                model=self.model,
                input_text=conversation[:200] + "..." if len(conversation) > 200 else conversation,
                ttft_ms=ttft_time,
                e2e_latency_ms=e2e_latency,
                tokens_per_second=tokens_per_second,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                start_time=start_time,
                end_time=end_time,
                tool_calls=bool(tool_calls),
                tool_call_count=tool_call_count,
                tool_call_latency_ms=tool_call_latency
            )
            
            return {
                "conversation": conversation,
                "messages": messages,
                "completion": full_response,
                "ttft_ms": ttft_time,
                "e2e_latency_ms": e2e_latency,
                "tokens_per_second": tokens_per_second,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tool_calls": tool_calls,
                "tool_call_count": tool_call_count,
                "tool_call_latency_ms": tool_call_latency
            }
            
        except Exception as e:
            logger.error(f"[GroqCloud] Error generating completion: {str(e)}", exc_info=True)
            raise

class AzureOpenAIClient:
    """Client for Azure OpenAI API."""
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str = "2024-08-01-preview",
        model: str = None,
        deployment_name: str = None,
        region: str = "eastus",
        max_tokens: int = 500,
        temperature: float = 0.7,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model or os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
        self.deployment_name = deployment_name or os.getenv('AZURE_OPENAI_DEPLOYMENT', self.model)
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize the client with appropriate timeout settings
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            timeout=timeout,
            max_retries=2,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=timeout,
                    write=5.0,
                    pool=None
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10
                ),
                trust_env=False
            )
        )
        
        logger.info(f"Initialized Azure OpenAI client for {self.model} (deployment: {self.deployment_name}) in {region}")

    def _parse_conversation_to_messages(self, conversation: str) -> List[Dict[str, str]]:
        """Parse a conversation string into a list of messages for the API."""
        lines = conversation.strip().split('\n')
        messages = [{"role": "system", "content": INTERVIEW_SYSTEM_PROMPT}]
        
        current_speaker = None
        current_message = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Interviewer:"):
                if current_speaker:
                    messages.append({"role": current_speaker, "content": current_message.strip()})
                current_speaker = "user"
                current_message = line.replace("Interviewer:", "").strip()
            elif line.startswith("Candidate:"):
                if current_speaker:
                    messages.append({"role": current_speaker, "content": current_message.strip()})
                current_speaker = "user"
                current_message = f"[Candidate's response]: {line.replace('Candidate:', '').strip()}"
            else:
                current_message += " " + line
        
        if current_speaker and current_message:
            messages.append({"role": current_speaker, "content": current_message.strip()})
        
        return messages

    async def generate_completion(self, conversation: str, use_tools: bool = True, stream: bool = False) -> Dict[str, Any]:
        """Generate a completion for the given conversation and measure latency metrics."""
        try:
            # Initial setup
            start_time = time.time()
            ttft_time = None
            first_token_time = None
            tool_call_count = 0
            tool_call_latency = 0.0
            full_response = ""
            tool_calls = []
            chunk_count = 0
            
            # Parse messages before starting latency measurement
            messages = self._parse_conversation_to_messages(conversation)
            
            try:
                if stream:
                    # Streaming mode
                    response_stream = await self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stream=True,
                        tools=TOOLS if use_tools else None
                    )
                    
                    async for chunk in response_stream:
                        chunk_count += 1
                        if not chunk or not chunk.choices:
                            continue
                        
                        # Record time to first token
                        if not first_token_time and chunk.choices and (
                            (hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content) or 
                            (hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls)
                        ):
                            first_token_time = time.time()
                            ttft_time = (first_token_time - start_time) * 1000
                        
                        # Handle tool calls
                        if chunk.choices and hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                if hasattr(tool_call, 'function'):
                                    tool_call_data = {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                    tool_calls.append(tool_call_data)
                                    tool_call_count += 1
                        
                        # Accumulate response
                        if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                else:
                    # Non-streaming mode
                    response = await self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stream=False,
                        tools=TOOLS if use_tools else None
                    )
                    
                    # Record TTFT (in non-streaming mode, this is the same as total latency)
                    first_token_time = time.time()
                    ttft_time = (first_token_time - start_time) * 1000
                    
                    # Process response
                    if response.choices:
                        # Handle tool calls
                        if hasattr(response.choices[0], 'tool_calls') and response.choices[0].tool_calls:
                            for tool_call in response.choices[0].tool_calls:
                                tool_call_data = {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.arguments
                                }
                                tool_calls.append(tool_call_data)
                                tool_call_count += 1
                        
                        # Get content
                        if hasattr(response.choices[0], 'message') and response.choices[0].message.content:
                            full_response = response.choices[0].message.content
            
            except Exception as api_error:
                logger.error(f"[Azure] API error: {str(api_error)}")
                if hasattr(api_error, 'response'):
                    logger.error(f"Response status: {api_error.response.status_code}")
                    logger.error(f"Response body: {await api_error.response.text()}")
                raise
            
            # Process tool calls if any
            if tool_calls:
                tool_start_time = time.time()
                for tool_call in tool_calls:
                    if tool_call["name"] == "get_next_question":
                        next_question = get_next_question()
                        full_response += f"\nNext question: {next_question}"
                    elif tool_call["name"] == "repeat_question":
                        current_question = conversation.split("Interviewer:")[-1].split("Candidate:")[0].strip()
                        full_response += f"\nRepeating question: {current_question}"
                tool_call_latency = (time.time() - tool_start_time) * 1000
            
            end_time = time.time()
            e2e_latency = (end_time - start_time) * 1000
            
            # Calculate metrics
            try:
                input_tokens = len(conversation.split()) // 3 * 4
                output_tokens = len(full_response.split()) // 3 * 4
            except Exception as e:
                logger.warning(f"Could not get token counts: {str(e)}")
                input_tokens = len(conversation) // 4
                output_tokens = len(full_response) // 4
            
            tokens_per_second = 0
            if output_tokens > 0 and first_token_time:
                generation_time = (end_time - first_token_time)
                if generation_time > 0:
                    tokens_per_second = output_tokens / generation_time
            
            # Log everything after latency calculations are complete
            logger.info(f"[Azure] Completion Summary:")
            logger.info(f"[Azure] Input conversation: {conversation}")
            logger.info(f"[Azure] Messages sent:\n{json.dumps(messages, indent=2)}")
            logger.info(f"[Azure] Mode: {'tools' if use_tools else 'chat'}, Streaming: {stream}")
            if stream:
                logger.info(f"[Azure] Received {chunk_count} chunks")
            if tool_calls:
                logger.info(f"[Azure] Tool calls made ({len(tool_calls)}):\n{json.dumps(tool_calls, indent=2)}")
            logger.info(f"[Azure] Final response:\n{full_response}")
            logger.info(f"[Azure] Performance metrics:")
            logger.info(f"  - TTFT: {ttft_time:.2f}ms")
            logger.info(f"  - E2E Latency: {e2e_latency:.2f}ms")
            logger.info(f"  - Tokens/second: {tokens_per_second:.2f}")
            logger.info(f"  - Tool call latency: {tool_call_latency:.2f}ms")
            
            # Add metrics to collector
            llm_metrics.add_metric(
                service_name="Azure",
                region=self.region,
                model=self.model,
                input_text=conversation[:200] + "..." if len(conversation) > 200 else conversation,
                ttft_ms=ttft_time,
                e2e_latency_ms=e2e_latency,
                tokens_per_second=tokens_per_second,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                start_time=start_time,
                end_time=end_time,
                tool_calls=bool(tool_calls),
                tool_call_count=tool_call_count,
                tool_call_latency_ms=tool_call_latency
            )
            
            return {
                "conversation": conversation,
                "messages": messages,
                "completion": full_response,
                "ttft_ms": ttft_time,
                "e2e_latency_ms": e2e_latency,
                "tokens_per_second": tokens_per_second,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tool_calls": tool_calls,
                "tool_call_count": tool_call_count,
                "tool_call_latency_ms": tool_call_latency
            }
            
        except Exception as e:
            logger.error(f"[Azure] Error generating completion: {str(e)}", exc_info=True)
            raise

def calculate_latency_stats(metrics: List[LLMMetricEntry]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate summary statistics for latency metrics by service and region."""
    stats = {}
    
    for metric in metrics:
        service_region = f"{metric.service_name}/{metric.region}"
        if service_region not in stats:
            stats[service_region] = {
                'ttft': [],
                'e2e_latency': [],
                'tokens_per_second': [],
                'tool_call_latency': []
            }
        
        stats[service_region]['ttft'].append(metric.ttft_ms)
        stats[service_region]['e2e_latency'].append(metric.e2e_latency_ms)
        stats[service_region]['tokens_per_second'].append(metric.tokens_per_second)
        if metric.tool_calls:
            stats[service_region]['tool_call_latency'].append(metric.tool_call_latency_ms)
    
    summary = {}
    for service_region, metrics in stats.items():
        summary[service_region] = {}
        for metric_name, values in metrics.items():
            if values:  # Only calculate stats if we have values
                summary[service_region][metric_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'count': len(values)
                }
    
    return summary

def display_summary_stats(summary: Dict[str, Dict[str, Dict[str, float]]]):
    """Display summary statistics in a formatted way."""
    logger.info("\n=== Latency Summary (all times in ms) ===")
    # Then display the full stats if needed
    logger.info("\n=== Detailed Statistics ===")
    for service_region, metrics in summary.items():
        logger.info(f"\n{service_region}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            logger.info(f"    Mean: {stats['mean']:.2f}")
            logger.info(f"    Median: {stats['median']:.2f}")
            logger.info(f"    Min: {stats['min']:.2f}")
            logger.info(f"    Max: {stats['max']:.2f}")
            logger.info(f"    Std Dev: {stats['std']:.2f}")
            logger.info(f"    Sample Count: {stats['count']}")

    # In the end display the concise summary
    logger.info("\nEnd-to-End Latency Summary:")
    for service_region, metrics in summary.items():
        if 'e2e_latency' in metrics:
            stats = metrics['e2e_latency']
            logger.info(f"{service_region:<30} Mean: {stats['mean']:>8.2f} | Max: {stats['max']:>8.2f} | StdDev: {stats['std']:>8.2f} (n={stats['count']})")

# Add Groq model mapping
GROQ_MODEL_ALIASES = {
    'mixtral': 'mixtral-8x7b-32768',
    'specdec': 'llama-3.3-70b-specdec',  # Using llama3 for specialized/decontaminated tasks
    'versatile': 'llama-3.3-70b-versatile',  # Using  for versatile tasks
    'gemma': 'gemma-7b-it',
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LLM latency benchmarks')
    parser.add_argument('--services', type=str, nargs='+', default=['azure'],
                      choices=['azure', 'groq'],
                      help='Services to test: azure and/or groq (default: azure)')
    parser.add_argument('--regions', type=str, nargs='+', default=['eastus', 'westus', 'sweden', 'india'],
                      help='Azure regions to test (default: eastus westus sweden india). Only used for Azure service.')
    parser.add_argument('--model', type=str, default=None,
                      help='Model to test for Azure (default: from .env AZURE_OPENAI_MODEL)')
    parser.add_argument('--groq-models', type=str, nargs='+',
                      choices=list(GROQ_MODEL_ALIASES.keys()),
                      help='Groq models to test (e.g., mixtral llama2 specdec versatile gemma)')
    parser.add_argument('--deployment', type=str, default=None,
                      help='Deployment name for Azure OpenAI (default: from .env AZURE_OPENAI_DEPLOYMENT or model name)')
    parser.add_argument('--iterations', type=int, default=10,
                      help='Number of iterations to run (default: 10)')
    parser.add_argument('--wait-time', type=int, default=5,
                      help='Wait time between iterations in seconds (default: 5)')
    parser.add_argument('--save-responses', action='store_true',
                      help='Save full responses to disk (default: False)')
    parser.add_argument('--mode', type=str, choices=['tools', 'chat'], default='tools',
                      help='Mode to run: tools (with tool calling) or chat (without tools) (default: tools)')
    parser.add_argument('--stream', action='store_true', default=False,
                      help='Use streaming mode for responses (default: False)')
    return parser.parse_args()

def validate_env_vars(regions: List[str], services: List[str], groq_models: Optional[List[str]] = None):
    """Validate that necessary environment variables are set."""
    valid_regions = []
    
    for service in services:
        if service == 'groq':
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                if groq_models:
                    # Add a region entry for each Groq model
                    for model_alias in groq_models:
                        model_name = GROQ_MODEL_ALIASES[model_alias]
                        valid_regions.append(f"groq-{model_alias}")
                        logger.info(f"Added Groq model: {model_alias} -> {model_name}")
                else:
                    # Default to mixtral if no models specified
                    valid_regions.append("groq-mixtral")
                    logger.info("Using default Groq model: mixtral")
                logger.info("GroqCloud credentials found")
            else:
                logger.error("GroqCloud API key not found. Set GROQ_API_KEY environment variable.")
        
        elif service == 'azure':
            # Check for default Azure credentials
            default_key = os.getenv('AZURE_OPENAI_API_KEY')
            default_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            
            if not default_key or not default_endpoint:
                logger.warning("Default Azure OpenAI credentials not found")
            
            # Check region-specific credentials
            for region in regions:
                region_key = os.getenv(f'AZURE_OPENAI_API_KEY_{region.upper()}')
                region_endpoint = os.getenv(f'AZURE_OPENAI_ENDPOINT_{region.upper()}')
                
                if region_key and region_endpoint:
                    valid_regions.append(region)
                elif default_key and default_endpoint:
                    valid_regions.append(region)
                    logger.warning(f"Using default credentials for region {region}")
                else:
                    logger.error(f"No credentials available for region {region}, will be skipped")
    
    if not valid_regions:
        raise ValueError("No valid regions/credentials found. Check your environment variables.")
    
    return valid_regions

async def run_llm_test(
    regions: List[str],
    model: Union[str, Dict[str, str]],
    deployment_name: str,
    num_iterations: int = 10,
    wait_time: int = 5,
    save_responses: bool = False,
    use_tools: bool = True,
    use_groq: bool = False,
    stream: bool = False
):
    """Run LLM latency test across multiple regions."""
    logger.info(f"Starting LLM test with {num_iterations} iterations across regions: {regions}")
    logger.info(f"Mode: {'tools' if use_tools else 'chat'}, Streaming: {stream}")
    
    # Initialize conversation generator
    conversation_generator = InterviewConversationGenerator()
    
    # Create output directory for this test run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create responses directory if needed
    responses_dir = None
    if save_responses:
        responses_dir = os.path.join(run_dir, 'responses')
        os.makedirs(responses_dir, exist_ok=True)
        logger.info(f"Saving responses to {responses_dir}")
    
    # Initialize clients for each region/model
    clients = {}
    for region in regions:
        if region.startswith('groq-'):
            # Get Groq credentials
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logger.error("GroqCloud API key not found, skipping")
                continue
            
            # Extract model alias from region string
            model_alias = region.split('-')[1]
            model_name = GROQ_MODEL_ALIASES[model_alias]
            
            clients[region] = GroqCloudClient(
                api_key=api_key,
                model=model_name,
                region=region
            )
        else:
            # Get region-specific environment variables for Azure
            api_key = os.getenv(f'AZURE_OPENAI_API_KEY_{region.upper()}')
            endpoint = os.getenv(f'AZURE_OPENAI_ENDPOINT_{region.upper()}')
            
            if not api_key or not endpoint:
                logger.warning(f"Missing credentials for region {region}, using default credentials")
                api_key = os.getenv('AZURE_OPENAI_API_KEY')
                endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                
                # If still no credentials, skip this region
                if not api_key or not endpoint:
                    logger.error(f"No credentials available for region {region}, skipping")
                    continue
            
            # Get Azure model
            azure_model = model.get('azure', model) if isinstance(model, dict) else model
            clients[region] = AzureOpenAIClient(
                api_key=api_key,
                endpoint=endpoint,
                model=azure_model,
                deployment_name=deployment_name,
                region=region
            )
    
    if not clients:
        raise ValueError("No valid clients could be initialized. Check your environment variables.")
    
    # Main test loop
    for i in range(num_iterations):
        logger.info(f"Iteration {i+1}/{num_iterations}")
        conversation = conversation_generator.generate_conversation()
        
        # Create tasks for all regions
        tasks = []
        for region, client in clients.items():
            task = asyncio.create_task(test_region(
                region=region,
                client=client,
                conversation=conversation,
                output_dir=responses_dir if save_responses else None,
                iteration=i+1,
                use_tools=use_tools,
                stream=stream
            ))
            tasks.append(task)
        
        # Wait for all regions to complete
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in iteration {i+1}: {str(e)}")
        
        # Flush metrics every 5 iterations
        if (i + 1) % 5 == 0:
            flush_metrics_to_disk()
        
        # Wait between iterations
        if i < num_iterations - 1 and wait_time > 0:
            logger.info(f"Waiting {wait_time} seconds before next iteration...")
            await asyncio.sleep(wait_time)
    
    # Calculate and display summary statistics
    summary = calculate_latency_stats(llm_metrics.metrics)
    display_summary_stats(summary)
    
    # Final metrics flush
    flush_metrics_to_disk()
    logger.info("LLM test completed")

async def test_region(
    region: str,
    client: Union[GroqCloudClient, AzureOpenAIClient],
    conversation: str,
    output_dir: Optional[str] = None,
    iteration: int = 0,
    use_tools: bool = True,
    stream: bool = False
) -> None:
    """Test a single region with the given conversation."""
    try:
        logger.info(f"Testing region: {region}")
        result = await client.generate_completion(conversation, use_tools, stream)
        
        # Save response if requested
        if output_dir:
            timestamp = datetime.now().strftime('%H%M%S')
            region_dir = os.path.join(output_dir, region)
            os.makedirs(region_dir, exist_ok=True)
            response_file = os.path.join(region_dir, f'response_{iteration}_{timestamp}.json')
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved response to {response_file}")
        
    except Exception as e:
        logger.error(f"Error testing region {region}: {str(e)}")
        return None

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Validate environment variables and get valid regions
        valid_regions = validate_env_vars(args.regions, args.services, args.groq_models)
        
        if not valid_regions:
            logger.error("No valid regions found. Exiting.")
            return
        
        # Get model and deployment based on services
        if 'groq' in args.services:
            if args.groq_models:
                groq_models = {f"groq-{alias}": GROQ_MODEL_ALIASES[alias] for alias in args.groq_models}
                if 'azure' in args.services:
                    model = {
                        **groq_models,
                        'azure': args.model or os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
                    }
                else:
                    model = groq_models
            else:
                model = {'groq-mixtral': 'mixtral-8x7b-32768'}
                if 'azure' in args.services:
                    model['azure'] = args.model or os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
            
            deployment = args.deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT', model.get('azure', 'gpt-4'))
        else:
            model = args.model or os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
            deployment = args.deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT', model)
        
        # Run the async test
        asyncio.run(run_llm_test(
            regions=valid_regions,
            model=model,
            deployment_name=deployment,
            num_iterations=args.iterations,
            wait_time=args.wait_time,
            save_responses=args.save_responses,
            use_tools=args.mode == 'tools',
            use_groq='groq' in args.services,
            stream=args.stream
        ))
        
    except Exception as e:
        logger.error(f"Error running LLM test: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
