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
from openai import AsyncAzureOpenAI
import json
from pathlib import Path

# Load environment variables
load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
logger.add(
    f"log_llm_latency_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    level="DEBUG",
)

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

class LLMMetrics:
    """Class for collecting and saving LLM metrics."""
    def __init__(self):
        self.metrics: List[LLMMetricEntry] = []
    
    def add_metric(self, service_name: str, region: str, model: str, input_text: str, 
                  ttft_ms: float, e2e_latency_ms: float, tokens_per_second: float,
                  input_tokens: int, output_tokens: int, start_time: float, end_time: float):
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
            end_time=end_time
        ))
    
    def save_to_csv(self, filename: str):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Service', 'Region', 'Model', 'Input Text', 'TTFT (ms)', 
                           'E2E Latency (ms)', 'Tokens/Second', 'Input Tokens',
                           'Output Tokens', 'Start Time', 'End Time'])
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
                    metric.end_time
                ])

# Create global metrics collector
llm_metrics = LLMMetrics()

def flush_metrics_to_disk():
    """Save current metrics to CSV file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('data'):
        os.makedirs('data')
    llm_metrics.save_to_csv(f'data/llm_metrics_{timestamp}.csv')

# System prompt for interview assistant
INTERVIEW_SYSTEM_PROMPT = """
You are an AI interview assistant that helps analyze interview conversations between an interviewer and a candidate.
Your task is to determine the next action the interviewer should take based on the conversation history.

Possible actions:
1. MOVE_TO_NEXT_QUESTION - When the candidate has answered the current question sufficiently
2. REPEAT_QUESTION - When the candidate's answer is unclear or incomplete
3. ASK_FOR_EXAMPLES - When the candidate's answer lacks specific examples
4. PROBE_DEEPER - When the candidate's answer is good but could benefit from more depth
5. CLARIFY_ANSWER - When the candidate's answer contains contradictions or confusion

Analyze the conversation carefully, focusing on:
- Whether the candidate directly answered the question asked
- The completeness and clarity of the candidate's response
- Whether the candidate provided concrete examples when appropriate
- The depth and thoughtfulness of the candidate's answer

Provide your recommendation for the next action. If its clarification respond with clarification question. Always provide an action for next step. RESPOND IN JSON FORMAT.
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
                    connect=5.0,    # Connection timeout
                    read=timeout,   # Read timeout
                    write=5.0,      # Write timeout
                    pool=None       # No pool timeout needed
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
                # Save previous message if exists
                if current_speaker:
                    messages.append({"role": current_speaker, "content": current_message.strip()})
                
                current_speaker = "user"
                current_message = line.replace("Interviewer:", "").strip()
            elif line.startswith("Candidate:"):
                # Save previous message if exists
                if current_speaker:
                    messages.append({"role": current_speaker, "content": current_message.strip()})
                
                current_speaker = "user"  # We're treating candidate responses as user messages too
                current_message = f"[Candidate's response]: {line.replace('Candidate:', '').strip()}"
            else:
                # Continue previous message
                current_message += " " + line
        
        # Add the last message
        if current_speaker and current_message:
            messages.append({"role": current_speaker, "content": current_message.strip()})
        
        return messages

    async def generate_completion(self, conversation: str) -> Dict[str, Any]:
        """Generate a completion for the given conversation and measure latency metrics."""
        try:
            logger.info(f"Generating completion for conversation: {conversation[:50]}..." if len(conversation) > 50 else conversation)
            
            # Parse conversation into messages
            messages = self._parse_conversation_to_messages(conversation)
            
            start_time = time.time()
            ttft_recorded = False
            ttft_time = None
            first_token_time = None
            
            # Track tokens for calculating throughput
            input_tokens = 0
            output_tokens = 0
            
            # Use streaming for more accurate TTFT measurement
            stream = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            # Collect the response
            full_response = ""
            
            async for chunk in stream:
                # Record time to first token
                if not ttft_recorded and chunk.choices and chunk.choices[0].delta.content:
                    first_token_time = time.time()
                    ttft_time = (first_token_time - start_time) * 1000  # ms
                    ttft_recorded = True
                    logger.debug(f"First token received after {ttft_time:.2f}ms")
                
                # Accumulate response
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            # Record end time and calculate metrics
            end_time = time.time()
            e2e_latency = (end_time - start_time) * 1000  # ms
            logger.debug(f"Full response: {full_response}")
            
            # Get token counts from usage info (will be available in the last chunk)
            try:
                # For Azure OpenAI, we need to estimate token counts
                input_tokens = len(conversation.split()) // 3 * 4  # Rough estimate: 4 tokens per 3 words
                output_tokens = len(full_response.split()) // 3 * 4
            except Exception as e:
                logger.warning(f"Could not get token counts: {str(e)}")
                # Fallback estimates
                input_tokens = len(conversation) // 4
                output_tokens = len(full_response) // 4
            
            # Calculate tokens per second (if we have output tokens)
            tokens_per_second = 0
            if output_tokens > 0 and ttft_recorded:
                generation_time = (end_time - first_token_time)  # seconds
                if generation_time > 0:
                    tokens_per_second = output_tokens / generation_time
            
            # Record metrics
            llm_metrics.add_metric(
                service_name="AzureOpenAI",
                region=self.region,
                model=self.model,
                input_text=conversation[:200] + "..." if len(conversation) > 200 else conversation,
                ttft_ms=ttft_time,
                e2e_latency_ms=e2e_latency,
                tokens_per_second=tokens_per_second,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                start_time=start_time,
                end_time=end_time
            )
            
            logger.info(f"Completion generated. E2E latency: {e2e_latency:.2f}ms, TTFT: {ttft_time:.2f}ms")
            
            return {
                "conversation": conversation,
                "messages": messages,
                "completion": full_response,
                "ttft_ms": ttft_time,
                "e2e_latency_ms": e2e_latency,
                "tokens_per_second": tokens_per_second,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}", exc_info=True)
            raise

async def run_llm_test(
    regions: List[str],
    model: str,
    deployment_name: str,
    num_iterations: int = 10,
    wait_time: int = 5,
    save_responses: bool = False
):
    """Run LLM latency test across multiple regions.
    All regions are tested concurrently for the same prompt to ensure fair comparison."""
    logger.info(f"Starting LLM test with {num_iterations} iterations across regions: {regions}")
    
    # Initialize conversation generator
    conversation_generator = InterviewConversationGenerator()
    
    # Create output directory for responses if needed
    output_dir = None
    if save_responses:
        output_dir = os.path.join('data', 'output', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving responses to {output_dir}")
    
    # Initialize clients for each region
    clients = {}
    for region in regions:
        # Get region-specific environment variables
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
        
        clients[region] = AzureOpenAIClient(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
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
                output_dir=output_dir if save_responses else None,
                iteration=i+1
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
    
    # Final metrics flush
    flush_metrics_to_disk()
    logger.info("LLM test completed")

async def test_region(
    region: str,
    client: AzureOpenAIClient,
    conversation: str,
    output_dir: Optional[str] = None,
    iteration: int = 0
) -> None:
    """Test a single region with the given conversation."""
    try:
        logger.info(f"Testing region: {region}")
        result = await client.generate_completion(conversation)
        
        # Save response if requested
        if output_dir:
            timestamp = datetime.now().strftime('%H%M%S')
            response_file = os.path.join(output_dir, f'{region}_{iteration}_{timestamp}.json')
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved response to {response_file}")
        
    except Exception as e:
        logger.error(f"Error testing region {region}: {str(e)}")
        return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LLM latency benchmarks')
    parser.add_argument('--regions', type=str, nargs='+', default=['eastus', 'westus', 'sweden', 'india'],
                      help='Azure regions to test (default: eastus westus sweden india)')
    parser.add_argument('--model', type=str, default=None,
                      help='Model to test (default: from .env AZURE_OPENAI_MODEL or gpt-4)')
    parser.add_argument('--deployment', type=str, default=None,
                      help='Deployment name (default: from .env AZURE_OPENAI_DEPLOYMENT or model name)')
    parser.add_argument('--iterations', type=int, default=10,
                      help='Number of iterations to run (default: 10)')
    parser.add_argument('--wait-time', type=int, default=5,
                      help='Wait time between iterations in seconds (default: 5)')
    parser.add_argument('--save-responses', action='store_true',
                      help='Save full responses to disk (default: False)')
    return parser.parse_args()

def validate_env_vars(regions: List[str]):
    """Validate that necessary environment variables are set."""
    # Check for default credentials
    default_key = os.getenv('AZURE_OPENAI_API_KEY')
    default_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if not default_key or not default_endpoint:
        logger.warning("Default Azure OpenAI credentials not found")
    
    # Check region-specific credentials
    valid_regions = []
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
        raise ValueError("No valid regions with credentials found. Check your environment variables.")
    
    return valid_regions

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Validate environment variables
        valid_regions = validate_env_vars(args.regions)
        
        if not valid_regions:
            logger.error("No valid regions found. Exiting.")
            return
        
        # Get model and deployment from args or environment
        model = args.model or os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
        deployment = args.deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT', model)
        
        # Run the async test
        asyncio.run(run_llm_test(
            regions=valid_regions,
            model=model,
            deployment_name=deployment,
            num_iterations=args.iterations,
            wait_time=args.wait_time,
            save_responses=args.save_responses
        ))
        
    except Exception as e:
        logger.error(f"Error running LLM test: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
