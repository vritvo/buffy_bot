import os
import json
import tempfile
import toml
import shutil
import re
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from dotenv import load_dotenv
from anthropic import Anthropic

from zulip import Client
from grad_bot_logger import GradBotLogger

# Load environment variables
load_dotenv()

console = Console()


# Rate limiting state
_last_api_call_time = 0
_tokens_used_in_current_minute = 0
_minute_start_time = 0


def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text string.
    Uses a rough approximation of ~4 characters per token for English text.
    """
    return len(text) // 4


def wait_for_rate_limit(estimated_input_tokens: int, tokens_per_minute_limit: int = 25000, enabled: bool = True):
    """Wait if necessary to respect rate limits.
    
    Args:
        estimated_input_tokens: Estimated number of input tokens for the next request
        tokens_per_minute_limit: Maximum tokens allowed per minute (default: 25000)
        enabled: Whether rate limiting is enabled (default: True)
    """
    if not enabled:
        return
    
    global _last_api_call_time, _tokens_used_in_current_minute, _minute_start_time
    
    current_time = time.time()
    
    # Reset counter if we're in a new minute
    if current_time - _minute_start_time >= 60:
        _tokens_used_in_current_minute = 0
        _minute_start_time = current_time
    
    # Check if this request would exceed the limit
    if _tokens_used_in_current_minute + estimated_input_tokens > tokens_per_minute_limit:
        # Calculate how long to wait until the next minute
        time_until_next_minute = 60 - (current_time - _minute_start_time)
        if time_until_next_minute > 0:
            console.print(f"[yellow]Rate limit approached ({_tokens_used_in_current_minute + estimated_input_tokens}/{tokens_per_minute_limit} tokens)[/yellow]")
            console.print(f"[yellow]Waiting {time_until_next_minute:.1f} seconds for rate limit reset...[/yellow]")
            time.sleep(time_until_next_minute + 1)  # Add 1 second buffer
            # Reset after waiting
            _tokens_used_in_current_minute = 0
            _minute_start_time = time.time()
    
    # Track this request
    _tokens_used_in_current_minute += estimated_input_tokens
    _last_api_call_time = time.time()


def get_most_recent_grad_notes_folder() -> Path:
    """Get the most recent grad_notes folder based on timestamp in folder name
    
    Returns:
        Path to the most recent grad_notes folder
        
    Raises:
        FileNotFoundError if no grad_notes folders exist
    """
    grad_notes_dir = Path("grad_notes")
    if not grad_notes_dir.exists():
        raise FileNotFoundError("grad_notes directory not found")
    
    # Find all folders that match the grad_bot_analysis pattern
    grad_note_folders = [
        d for d in grad_notes_dir.iterdir() 
        if d.is_dir() and (d.name.startswith('grad_bot_analysis_') or d.name.startswith('nietzsche_') or d.name.startswith('neitzsche_'))
    ]
    
    if not grad_note_folders:
        raise FileNotFoundError("No grad bot analysis folders found in grad_notes/")
    
    # Sort by modification time (most recent first)
    grad_note_folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    most_recent = grad_note_folders[0]
    console.print(f"[blue]Using most recent grad notes folder: {most_recent}[/blue]")
    
    return most_recent


def create_paper_folder(topic_shorthand: str = None, notes_folder: str = None) -> Path:
    """Create a topic-specific folder in the papers directory with consistent naming
    
    Args:
        topic_shorthand: Short identifier for the topic 
        notes_folder: Optional path to grad notes folder to extract shorthand from
    
    Returns:
        Path to the created folder
    """
    papers_dir = Path("papers")
    papers_dir.mkdir(exist_ok=True)
    
    # Determine shorthand from either parameter or notes folder name
    if topic_shorthand:
        # Clean the shorthand to be filesystem-safe
        safe_shorthand = "".join(c for c in topic_shorthand if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_shorthand = safe_shorthand.replace(' ', '_')
    elif notes_folder:
        # Extract shorthand from notes folder name (e.g., "nietzsche_20250820_171124" -> "nietzsche")
        notes_path = Path(notes_folder)
        folder_name = notes_path.name
        # Take everything before the first timestamp-like pattern
        safe_shorthand = re.split(r'_\d{8}_\d{6}', folder_name)[0]
    else:
        # Fallback: use a generic identifier
        safe_shorthand = "paper_topic"
    
    # Create topic-specific folder with format: <timestamp>_<shorthand>
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{timestamp}_{safe_shorthand}"
    folder_path = papers_dir / folder_name
    folder_path.mkdir(exist_ok=True)
    
    return folder_path


@dataclass
class Message:
    """Simple message data structure"""
    content: str
    sender: str
    timestamp: int
    message_id: int


class ZulipExtractor:
    """Extracts conversations from Zulip"""
    
    def __init__(self, api_key: str, site_url: str, email: str = None):
        # Create a temporary config file for the Zulip client
        # Use provided email, or try to extract from API key, or get from environment
        if email:
            user_email = email
        elif ":" in api_key:
            user_email = api_key.split(":")[0]
        else:
            # Try to get email from environment variable
            user_email = os.getenv('ZULIP_EMAIL')
            if not user_email:
                user_email = "user@example.com"  # Last resort placeholder
        
        config_content = f"""[api]
email = {user_email}
api_key = {api_key}
site = {site_url}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rc', delete=False) as f:
            f.write(config_content)
            config_file = f.name
        
        # Initialize Zulip client with the temporary config file
        self.client = Client(config_file=config_file)
        self.config_file = config_file  # Store for cleanup
        self.output_dir = Path("conversations")
        self.output_dir.mkdir(exist_ok=True)
    
    def __del__(self):
        # Clean up temporary config file
        if hasattr(self, 'config_file') and os.path.exists(self.config_file):
            try:
                os.unlink(self.config_file)
            except:
                pass
    
    def get_user_info(self, user_id: int) -> str:
        """Get username from user ID"""
        try:
            result = self.client.get_user_by_id(user_id)
            if result['result'] == 'success':
                return result['user']['full_name']
            return f"User_{user_id}"
        except Exception:
            return f"User_{user_id}"
    
    def extract_stream_messages(self, stream_name: str, topic: str, limit: int = 1000) -> List[Message]:
        """Extract messages from a specific stream and topic"""
        messages = []
        
        # Get stream ID
        streams = self.client.get_streams()
        stream_id = None
        
        if streams['result'] == 'success':
            for stream in streams['streams']:
                if stream['name'].lower() == stream_name.lower():
                    stream_id = stream['stream_id']
                    break
        
        if stream_id is None:
            raise ValueError(f"Stream '{stream_name}' not found")
        
        # Get messages from the topic
        request = {
            'anchor': 'newest',
            'num_before': limit,
            'num_after': 0,
            'narrow': [
                {'operator': 'stream', 'operand': stream_name},
                {'operator': 'topic', 'operand': topic}
            ]
        }
        
        result = self.client.get_messages(request)
        
        if result['result'] == 'success':
            for msg in result['messages']:
                # Skip system messages, file uploads, etc.
                if msg['type'] != 'stream':
                    continue
                
                # Skip messages with no content
                if not msg.get('content', '').strip():
                    continue
                
                message = Message(
                    content=msg['content'],
                    sender=msg.get('sender_full_name', 'Unknown'),
                    timestamp=msg['timestamp'],
                    message_id=msg['id']
                )
                messages.append(message)
        
        return messages
    
    def extract_private_messages(self, user_emails: List[str], limit: int = 1000) -> List[Message]:
        """Extract messages from a private conversation using email addresses"""
        messages = []
        
        # Zulip API has a maximum limit of 5000 messages per request
        max_batch_size = 5000
        
        if limit <= max_batch_size:
            # Single request for smaller limits
            return self._extract_private_messages_batch(user_emails, limit, 'newest')
        else:
            # Multiple requests for larger limits
            console.print(f"[yellow]Requesting {limit} messages in batches (max {max_batch_size} per batch)[/yellow]")
            
            remaining = limit
            anchor = 'newest'
            
            while remaining > 0:
                batch_size = min(remaining, max_batch_size)
                console.print(f"[blue]Fetching batch of {batch_size} messages (anchor: {anchor})[/blue]")
                
                batch_messages = self._extract_private_messages_batch(user_emails, batch_size, anchor)
                
                if not batch_messages:
                    console.print("[yellow]No more messages found, stopping batch retrieval[/yellow]")
                    break
                
                messages.extend(batch_messages)
                remaining -= len(batch_messages)
                
                # Update anchor to the oldest message ID in this batch for the next request
                if batch_messages:
                    # Sort messages by message ID to find the oldest (smallest ID)
                    sorted_messages = sorted(batch_messages, key=lambda m: m.message_id)
                    oldest_message = sorted_messages[0]
                    # For the next batch, we want messages older than this one
                    # Use the message ID as anchor
                    anchor = oldest_message.message_id
                    console.print(f"[blue]Retrieved {len(batch_messages)} messages, {remaining} remaining. Next anchor: {anchor}[/blue]")
                
                # If we got fewer messages than requested, we've reached the end
                if len(batch_messages) < batch_size:
                    console.print("[blue]Reached end of conversation[/blue]")
                    break
            
            console.print(f"[green]Total messages retrieved: {len(messages)}[/green]")
            return messages
    
    def _extract_private_messages_batch(self, user_emails: List[str], limit: int, anchor) -> List[Message]:
        """Extract a single batch of private messages"""
        messages = []
        
        # For private messages, we can use the email directly in the narrow
        # The Zulip API accepts email addresses for pm-with operator
        if anchor == 'newest':
            request = {
                'anchor': 'newest',
                'num_before': limit,
                'num_after': 0,
                'narrow': [
                    {'operator': 'pm-with', 'operand': ','.join(user_emails)}
                ]
            }
        else:
            # For subsequent batches, use the message ID anchor to get older messages
            # We want messages BEFORE this message ID (older messages)
            request = {
                'anchor': anchor - 1,  # Subtract 1 to avoid including the anchor message itself
                'num_before': limit,
                'num_after': 0,
                'narrow': [
                    {'operator': 'pm-with', 'operand': ','.join(user_emails)}
                ]
            }
        
        result = self.client.get_messages(request)
        
        if result['result'] == 'success':
            for msg in result['messages']:
                # Skip system messages, file uploads, etc.
                if msg['type'] != 'private':
                    continue
                
                # Skip messages with no content
                if not msg.get('content', '').strip():
                    continue
                
                message = Message(
                    content=msg['content'],
                    sender=msg.get('sender_full_name', 'Unknown'),
                    timestamp=msg['timestamp'],
                    message_id=msg['id']
                )
                messages.append(message)
        else:
            console.print(f"[red]Error in batch request: {result['msg']}[/red]")
        
        return messages
    
    def save_messages(self, messages: List[Message], filename: str):
        """Save messages to both JSON and TXT files"""
        if not messages:
            console.print("[yellow]No messages to save[/yellow]")
            return
        
        # Create output data structure for JSON
        output_data = {
            "metadata": {
                "extracted_at": datetime.now().isoformat(),
                "total_messages": len(messages),
                "first_message": min(msg.timestamp for msg in messages) if messages else None,
                "last_message": max(msg.timestamp for msg in messages) if messages else None
            },
            "messages": [asdict(msg) for msg in messages]
        }
        
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Save JSON file
        json_path = self.output_dir / filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Create TXT file with just the content
        txt_filename = filename.replace('.json', '.txt')
        txt_path = self.output_dir / txt_filename
        
        # Extract just the content and clean it up
        content_lines = []
        for msg in messages:
            content = msg.content.strip()
            if content:  # Only include non-empty content
                # Remove HTML tags if present (Zulip messages can have HTML)
                import re
                content = re.sub(r'<[^>]+>', '', content)
                
                # Replace quoted message headers (e.g., "NAME said:") with generic text
                # This replaces lines that match the pattern of "NAME said:" with "Quoting the following text:"
                content = re.sub(r'^[^:]+\s+said:\s*', 'Quoting the following text:\n', content, flags=re.MULTILINE)
                
                # Clean up any remaining empty lines at the start
                content = content.strip()
                
                if content:  # Only add if there's still content after cleaning
                    content_lines.append(content)
        
        # Save TXT file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(content_lines))
        
        console.print(f"[green]Saved {len(messages)} messages to {json_path}[/green]")
        console.print(f"[green]Saved {len(content_lines)} content blocks to {txt_path}[/green]")
        
        # Show file size comparison
        json_size = json_path.stat().st_size
        txt_size = txt_path.stat().st_size
        reduction = (1 - txt_size / json_size) * 100
        console.print(f"[blue]TXT file is {reduction:.1f}% smaller ({json_size:,} → {txt_size:,} bytes)[/blue]")
    
    def get_week_start(self, timestamp: int) -> datetime:
        """Get the start of the week (Sunday 3am ET) for a given timestamp"""
        # Convert timestamp to datetime in ET timezone
        et_tz = timezone(timedelta(hours=-5))  # EST (assuming no DST handling needed)
        dt = datetime.fromtimestamp(timestamp, tz=et_tz)
        
        # Find the most recent Sunday at 3am ET
        days_since_sunday = dt.weekday() + 1  # Monday=0, so Sunday=6, we want Sunday=0
        if days_since_sunday == 7:  # Already Sunday
            if dt.hour >= 3:  # After 3am, so this is current week
                days_since_sunday = 0
            else:  # Before 3am, so this is previous week
                days_since_sunday = 7
        
        # Calculate the Sunday at 3am
        week_start = dt.replace(hour=3, minute=0, second=0, microsecond=0)
        week_start = week_start - timedelta(days=days_since_sunday)
        
        return week_start
    
    def split_into_weekly_chunks(self, messages: List[Message], base_filename: str):
        """Split messages into weekly chunks and save as separate TXT files"""
        if not messages:
            console.print("[yellow]No messages to split[/yellow]")
            return
        
        # Group messages by week
        weekly_chunks = {}
        
        for msg in messages:
            week_start = self.get_week_start(msg.timestamp)
            week_key = week_start.strftime('%Y-%m-%d')
            
            if week_key not in weekly_chunks:
                weekly_chunks[week_key] = {
                    'week_start': week_start,
                    'messages': []
                }
            
            weekly_chunks[week_key]['messages'].append(msg)
        
        # Create weekly directory
        weekly_dir = self.output_dir / "weekly"
        weekly_dir.mkdir(exist_ok=True)
        
        # Save each week as a separate file
        base_name = base_filename.replace('.json', '').replace('.txt', '')
        
        console.print(f"[blue]Splitting into {len(weekly_chunks)} weekly chunks...[/blue]")
        
        for week_key, chunk_data in sorted(weekly_chunks.items()):
            week_start = chunk_data['week_start']
            week_messages = chunk_data['messages']
            
            # Create filename for this week
            week_filename = f"{base_name}_week_{week_key}.txt"
            week_path = weekly_dir / week_filename
            
            # Extract content and clean it up
            content_lines = []
            for msg in week_messages:
                content = msg.content.strip()
                if content:
                    # Remove HTML tags if present
                    import re
                    content = re.sub(r'<[^>]+>', '', content)
                    
                    # Replace quoted message headers (e.g., "NAME said:") with generic text
                    # This replaces lines that match the pattern of "NAME said:" with "Quoting the following text:"
                    content = re.sub(r'^[^:]+\s+said:\s*', 'Quoting the following text:\n', content, flags=re.MULTILINE)
                    
                    # Clean up any remaining empty lines at the start
                    content = content.strip()
                    
                    if content:  # Only add if there's still content after cleaning
                        content_lines.append(content)
            
            # Add week header
            week_end = week_start + timedelta(days=7)
            header = f"=== WEEK OF {week_start.strftime('%B %d, %Y')} - {week_end.strftime('%B %d, %Y')} ===\n"
            header += f"Messages: {len(week_messages)}\n"
            header += f"Content blocks: {len(content_lines)}\n\n"
            
            # Save the weekly file
            with open(week_path, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write('\n\n'.join(content_lines))
            
            console.print(f"[green]Week {week_key}: {len(week_messages)} messages → {week_path}[/green]")
        
        console.print(f"[blue]Weekly chunks saved to: {weekly_dir}[/blue]")


class PaperGenerator:
    """Generates academic papers from conversation transcripts using Claude"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.output_dir = Path("papers")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_system_prompt(self) -> tuple[str, str]:
        """Load system and user prompts from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        prompt_type = "buffy"
        if prompt_type not in config:
            available = list(config.keys())
            raise ValueError(f"Prompt type '{prompt_type}' not found. Available: {available}")
        
        prompt_config = config[prompt_type]
        system_prompt = prompt_config["system_prompt"]
        user_prompt = prompt_config["user_prompt"]
        
        return system_prompt, user_prompt
    
    def load_model_config(self) -> str:
        """Load paper generation model from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'models' not in config or 'paper_generation' not in config['models']:
            # Fallback to default model if not configured
            return "claude-sonnet-4-20250514"
        
        return config['models']['paper_generation']
    
    def load_conversation(self, conversation_file: str) -> str:
        """Load conversation from TXT file for Claude"""
        file_path = Path(conversation_file)
        
        # If user provided a JSON file, automatically switch to TXT
        if file_path.suffix == '.json':
            txt_path = file_path.with_suffix('.txt')
            if txt_path.exists():
                console.print(f"[yellow]Switching from JSON to TXT file for better token efficiency: {txt_path}[/yellow]")
                file_path = txt_path
            else:
                raise FileNotFoundError(f"TXT file not found: {txt_path}. Please run extract-private again to generate both files.")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Conversation file not found: {conversation_file}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError("No content found in conversation file")
        
        # Format for Claude with simple header and footer
        transcript = f"""=== CHAT TRANSCRIPT ===

{content}

=== END TRANSCRIPT ==="""
        
        return transcript
    
    def generate_paper(self, conversation_file: str, topic: str) -> str:
        """Generate a paper using Claude based on the conversation and topic"""
        
        console.print(f"[blue]Loading conversation from: {conversation_file}[/blue]")
        transcript = self.load_conversation(conversation_file)
        
        console.print(f"[blue]Loading system prompt[/blue]")
        system_prompt, user_prompt_template = self.load_system_prompt()
        
        console.print(f"[blue]Loading model configuration[/blue]")
        model = self.load_model_config()
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{CHAT_TRANSCRIPT}}", transcript)
        user_prompt = user_prompt.replace("{{SCRIPT_TRANSCRIPTS}}", "")  # No scripts for direct approach
        user_prompt = user_prompt.replace("{{PAPER_TOPIC}}", topic)
        
        # Estimate token count and wait for rate limit if necessary (disabled by default for legacy direct approach)
        estimated_input_tokens = estimate_token_count(system_prompt + user_prompt)
        console.print(f"[blue]Estimated input tokens: ~{estimated_input_tokens:,}[/blue]")
        wait_for_rate_limit(estimated_input_tokens, tokens_per_minute_limit=25000, enabled=False)
        
        console.print(f"[yellow]Sending request to Claude using model: {model}[/yellow]")
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            paper_content = response.content[0].text
            console.print(f"[green]✓ Paper generated ({len(paper_content)} characters)[/green]")
            return paper_content
            
        except Exception as e:
            raise Exception(f"Failed to generate paper with Claude: {e}")
    
    def save_paper(self, content: str, topic: str, conversation_file: str, topic_shorthand: str = None, model: str = None, paper_folder: Path = None) -> Path:
        """Save generated paper to a file in the appropriate folder structure"""
        # Use provided paper folder or create one using the shared function
        if paper_folder is None:
            paper_folder = create_paper_folder(topic_shorthand)
            console.print(f"[blue]Created paper folder: {paper_folder}[/blue]")
        
        # Simple filename - just "paper.md" since we're in a topic-specific folder
        filename = "paper.md"
        output_path = paper_folder / filename
        
        # Add metadata header to the paper
        if model is None:
            model = self.load_model_config()
        
        metadata = f"""---
title: "{topic}"
source_conversation: "{conversation_file}"
generated_at: "{datetime.now().isoformat()}"
model: "{model}"
---

"""
        
        full_content = metadata + content
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return output_path
    
    def generate_paper_from_notes(self, notes_folder: str, topic: str) -> str:
        """Generate a paper using Claude based on filtered grad bot notes"""
        
        notes_path = Path(notes_folder)
        if not notes_path.exists():
            raise FileNotFoundError(f"Notes folder not found: {notes_folder}")
        
        console.print(f"[blue]Loading grad bot notes from: {notes_folder}[/blue]")
        
        # Find all markdown files in the notes folder
        note_files = sorted([f for f in notes_path.glob("*.md")])
        
        if not note_files:
            raise FileNotFoundError(f"No markdown note files found in: {notes_folder}")
        
        console.print(f"[blue]Found {len(note_files)} note files[/blue]")
        
        # Combine all the filtered conversations and notes
        combined_content = []
        combined_content.append("=== FILTERED CONVERSATION NOTES ===\n")
        
        for note_file in note_files:
            with open(note_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Extract just the content after the metadata header
            if "---" in content:
                # Split on the second occurrence of "---" to skip the metadata
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    main_content = parts[2].strip()
                else:
                    main_content = content
            else:
                main_content = content
            
            combined_content.append(f"\n=== WEEK: {note_file.stem} ===\n")
            combined_content.append(main_content)
            combined_content.append("\n" + "="*50 + "\n")
        
        combined_content.append("\n=== END FILTERED NOTES ===")
        transcript = "\n".join(combined_content)
        
        console.print(f"[blue]Loading system prompt[/blue]")
        system_prompt, user_prompt_template = self.load_system_prompt()
        
        console.print(f"[blue]Loading model configuration[/blue]")
        model = self.load_model_config()
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{CHAT_TRANSCRIPT}}", transcript)
        user_prompt = user_prompt.replace("{{SCRIPT_TRANSCRIPTS}}", "")  # No scripts for this legacy method
        user_prompt = user_prompt.replace("{{PAPER_TOPIC}}", topic)
        
        console.print(f"[yellow]Sending request to Claude using model: {model}[/yellow]")
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            paper_content = response.content[0].text
            console.print(f"[green]✓ Paper generated from notes ({len(paper_content)} characters)[/green]")
            return paper_content
            
        except Exception as e:
            raise Exception(f"Failed to generate paper from notes with Claude: {e}")


class GradBot:
    """Graduate student research assistant for analyzing weekly conversation chunks"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.output_dir = Path("grad_notes")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = GradBotLogger()
    
    def load_grad_bot_prompts(self) -> tuple[str, str]:
        """Load grad bot system and user prompts from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        prompt_type = "grad_bot_buffy"
        if prompt_type not in config:
            available = [k for k in config.keys() if k.startswith('grad_bot_')]
            raise ValueError(f"Grad bot prompt type '{prompt_type}' not found. Available: {available}")
        
        prompt_config = config[prompt_type]
        system_prompt = prompt_config["system_prompt"]
        
        # Check if user_prompt exists, otherwise use a default
        if "user_prompt" in prompt_config:
            user_prompt = prompt_config["user_prompt"]
        else:
            # Fallback for old-style prompts that don't have separate user_prompt
            user_prompt = """Please analyze the following weekly conversation transcript:

{{CONVERSATION_TRANSCRIPT}}

Remember to maintain all important details while making it more concise and academic in tone."""
        
        return system_prompt, user_prompt
    
    def load_grad_bot_model_config(self) -> str:
        """Load grad bot model from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'models' not in config or 'grad_bot' not in config['models']:
            # Fallback to default model if not configured
            return "claude-sonnet-4-20250514"
        
        return config['models']['grad_bot']
    
    def load_grad_bot_api_settings(self) -> dict:
        """Load grad bot API settings from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        
        return {
            'temperature': config['api_settings']['grad_bot_temperature'],
            'max_tokens': config['api_settings']['grad_bot_max_tokens']
        }
    
    def analyze_weekly_chunk(self, weekly_file: str) -> str:
        """Analyze a weekly conversation chunk and extract relevant content"""
        
        file_path = Path(weekly_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Weekly file not found: {weekly_file}")
        
        # Load the weekly conversation
        with open(file_path, 'r', encoding='utf-8') as f:
            weekly_content = f.read().strip()
        
        if not weekly_content:
            raise ValueError("No content found in weekly file")
        
        console.print(f"[blue]Analyzing: {file_path.name}[/blue]")
        
        # Load system and user prompts
        system_prompt, user_prompt_template = self.load_grad_bot_prompts()
        
        # Load model configuration and API settings
        model = self.load_grad_bot_model_config()
        api_settings = self.load_grad_bot_api_settings()
        
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{CONVERSATION_TRANSCRIPT}}", weekly_content)
        
        # Estimate token count and wait for rate limit if necessary (disabled by default for grad bot)
        estimated_input_tokens = estimate_token_count(system_prompt + user_prompt)
        wait_for_rate_limit(estimated_input_tokens, tokens_per_minute_limit=25000, enabled=False)
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=api_settings['max_tokens'],
                temperature=api_settings['temperature'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            analysis = response.content[0].text
            console.print(f"[green]✓ Analysis complete ({len(analysis)} characters)[/green]")
            return analysis
            
        except Exception as e:
            raise Exception(f"Failed to analyze weekly chunk with Claude: {e}")
    
    def save_grad_notes(self, analysis: str, weekly_file: str, topic_folder_path: Path = None, model: str = None) -> Path:
        """Save grad bot analysis to a file in a topic-specific folder"""
        # Extract week identifier from filename
        week_file = Path(weekly_file)
        week_name = week_file.stem  # e.g., "private_conversation_week_2024-01-07"
        
        # Extract just the week part for the filename (e.g., "week_2024-01-07")
        if "week_" in week_name:
            week_part = week_name.split("week_", 1)[1]  # Get everything after "week_"
        else:
            week_part = week_name
        
        # Use provided folder or create a new one
        if topic_folder_path is None:
            # Create generic folder for grad bot analysis
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            topic_folder_name = f"grad_bot_analysis_{timestamp}"
            topic_folder_path = self.output_dir / topic_folder_name
            topic_folder_path.mkdir(exist_ok=True)
        
        # Filename is just the week identifier
        filename = f"{week_part}.md"
        
        output_path = topic_folder_path / filename
        
        # Add metadata header
        if model is None:
            model = self.load_grad_bot_model_config()
        
        metadata = f"""---
source_week: "{weekly_file}"
analyzed_at: "{datetime.now().isoformat()}"
model: "{model}"
grad_bot_type: "research_assistant"
---

# Grad Bot Analysis: {week_name}

**Source:** {weekly_file}

---

{analysis}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(metadata)
        
        return output_path

    def process_all_weekly_chunks(self, weekly_dir: str):
        """Process all weekly chunks in a directory sequentially"""
        weekly_path = Path(weekly_dir)
        if not weekly_path.exists():
            raise FileNotFoundError(f"Weekly directory not found: {weekly_dir}")
        
        # Find all weekly .txt files
        weekly_files = sorted([f for f in weekly_path.glob("*_week_*.txt")])
        
        if not weekly_files:
            console.print(f"[yellow]No weekly files found in {weekly_dir}[/yellow]")
            return []
        
        console.print(f"[blue]Found {len(weekly_files)} weekly files to process[/blue]")
        
        # Create folder for this grad bot run
        # Use generic naming since we don't know the topic yet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        topic_folder_name = f"grad_bot_analysis_{timestamp}"
        topic_folder_path = self.output_dir / topic_folder_name
        topic_folder_path.mkdir(exist_ok=True)
        
        console.print(f"[blue]Creating topic folder: {topic_folder_path}[/blue]")
        
        # Log the grad bot execution with prompts and settings
        system_prompt, user_prompt_template = self.load_grad_bot_prompts()
        model = self.load_grad_bot_model_config()
        api_settings = self.load_grad_bot_api_settings()
        
        prompt_hash = self.logger.log_grad_bot_execution(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            model=model,
            temperature=api_settings['temperature'],
            max_tokens=api_settings['max_tokens'],
            result_folder_name=topic_folder_name
        )
        
        console.print(f"[blue]Logged grad bot run with prompt hash: {prompt_hash}[/blue]")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for i, weekly_file in enumerate(weekly_files, 1):
                task = progress.add_task(f"Processing week {i}/{len(weekly_files)}: {weekly_file.name}", total=None)
                
                try:
                    # Analyze the weekly chunk
                    analysis = self.analyze_weekly_chunk(str(weekly_file))
                    
                    # Get model for metadata
                    model = self.load_grad_bot_model_config()
                    
                    # Save the analysis to the shared topic folder
                    output_path = self.save_grad_notes(analysis, str(weekly_file), topic_folder_path, model)
                    
                    results.append({
                        'week_file': str(weekly_file),
                        'analysis_file': str(output_path),
                        'success': True
                    })
                    
                    progress.update(task, description=f"✓ Completed: {weekly_file.name}")
                    
                except Exception as e:
                    console.print(f"[red]Error processing {weekly_file.name}: {e}[/red]")
                    results.append({
                        'week_file': str(weekly_file),
                        'error': str(e),
                        'success': False
                    })
                    
                    progress.update(task, description=f"✗ Failed: {weekly_file.name}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        console.print(f"\n[green]Grad bot analysis complete![/green]")
        console.print(f"[blue]Successfully processed: {successful}/{len(weekly_files)} weeks[/blue]")
        console.print(f"[blue]Notes saved to: {topic_folder_path}[/blue]")
        
        return results

    def process_specific_weekly_chunk(self, weekly_dir: str, specific_week: str, output_folder: str = None):
        """Process a specific weekly chunk and save to an existing or new folder"""
        weekly_path = Path(weekly_dir)
        if not weekly_path.exists():
            raise FileNotFoundError(f"Weekly directory not found: {weekly_dir}")
        
        # Find the specific weekly file
        weekly_file_pattern = f"*_week_{specific_week}.txt"
        weekly_files = list(weekly_path.glob(weekly_file_pattern))
        
        if not weekly_files:
            console.print(f"[red]No weekly file found for week {specific_week} in {weekly_dir}[/red]")
            console.print(f"[yellow]Looking for pattern: {weekly_file_pattern}[/yellow]")
            # Show available weeks
            available_weeks = []
            for f in weekly_path.glob("*_week_*.txt"):
                week_part = f.stem.split("week_", 1)[1] if "week_" in f.stem else "unknown"
                available_weeks.append(week_part)
            if available_weeks:
                console.print(f"[blue]Available weeks: {', '.join(sorted(set(available_weeks)))}[/blue]")
            return []
        
        weekly_file = weekly_files[0]  # Should only be one match
        console.print(f"[blue]Found weekly file: {weekly_file}[/blue]")
        
        # Determine output folder
        if output_folder:
            # Use existing folder
            topic_folder_path = Path(output_folder)
            if not topic_folder_path.exists():
                raise FileNotFoundError(f"Output folder not found: {output_folder}")
            console.print(f"[blue]Using existing output folder: {topic_folder_path}[/blue]")
        else:
            # Create new folder for this grad bot run
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            topic_folder_name = f"grad_bot_analysis_{timestamp}"
            topic_folder_path = self.output_dir / topic_folder_name
            topic_folder_path.mkdir(exist_ok=True)
            console.print(f"[blue]Created new output folder: {topic_folder_path}[/blue]")
        
        # Log the grad bot execution with prompts and settings (only if creating new folder)
        if not output_folder:
            system_prompt, user_prompt_template = self.load_grad_bot_prompts()
            model = self.load_grad_bot_model_config()
            api_settings = self.load_grad_bot_api_settings()
            
            prompt_hash = self.logger.log_grad_bot_execution(
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                model=model,
                temperature=api_settings['temperature'],
                max_tokens=api_settings['max_tokens'],
                result_folder_name=topic_folder_path.name
            )
            
            console.print(f"[blue]Logged grad bot run with prompt hash: {prompt_hash}[/blue]")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Processing week: {weekly_file.name}", total=None)
            
            try:
                # Analyze the weekly chunk
                analysis = self.analyze_weekly_chunk(str(weekly_file))
                
                # Get model for metadata
                model = self.load_grad_bot_model_config()
                
                # Save the analysis to the specified folder
                output_path = self.save_grad_notes(analysis, str(weekly_file), topic_folder_path, model)
                
                results.append({
                    'week_file': str(weekly_file),
                    'analysis_file': str(output_path),
                    'success': True
                })
                
                progress.update(task, description=f"✓ Completed: {weekly_file.name}")
                
            except Exception as e:
                console.print(f"[red]Error processing {weekly_file.name}: {e}[/red]")
                results.append({
                    'week_file': str(weekly_file),
                    'error': str(e),
                    'success': False
                })
                
                progress.update(task, description=f"✗ Failed: {weekly_file.name}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        console.print(f"\n[green]Grad bot analysis complete![/green]")
        console.print(f"[blue]Successfully processed: {successful}/1 week[/blue]")
        console.print(f"[blue]Notes saved to: {topic_folder_path}[/blue]")
        
        return results


class PostdocBot:
    """Postdoc bot for rating the relevance of grad student notes to a paper topic"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def load_postdoc_prompts(self) -> tuple[str, str]:
        """Load postdoc system and user prompts from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'postdoc_bot' not in config:
            raise ValueError("Postdoc bot prompt configuration not found in prompts.toml")
        
        prompt_config = config['postdoc_bot']
        system_prompt = prompt_config["system_prompt"]
        user_prompt = prompt_config["user_prompt"]
        
        return system_prompt, user_prompt
    
    def load_postdoc_model_config(self) -> str:
        """Load postdoc model from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'models' not in config or 'grad_bot' not in config['models']:
            # Use grad_bot model as default for postdoc
            return "claude-sonnet-4-20250514"
        
        return config['models']['grad_bot']
    
    def load_postdoc_api_settings(self) -> dict:
        """Load postdoc API settings from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        
        # Use research assistant settings as they're similar in scope
        return {
            'temperature': config['api_settings']['research_assistant_temperature'],
            'max_tokens': config['api_settings']['research_assistant_max_tokens']
        }
    
    def load_weekly_note(self, note_file: Path) -> str:
        """Load a single weekly grad bot note file"""
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Extract just the content after the metadata header
        if "---" in content:
            # Split on the second occurrence of "---" to skip the metadata
            parts = content.split("---", 2)
            if len(parts) >= 3:
                main_content = parts[2].strip()
            else:
                main_content = content
        else:
            main_content = content
        
        return main_content
    
    def rate_weekly_note(self, note_file: Path, topic: str) -> int:
        """Rate a single weekly note file for relevance to the topic"""
        console.print(f"[blue]Rating: {note_file.name}[/blue]")
        
        # Load the weekly note content
        weekly_content = self.load_weekly_note(note_file)
        
        # Load prompts and settings
        system_prompt, user_prompt_template = self.load_postdoc_prompts()
        model = self.load_postdoc_model_config()
        api_settings = self.load_postdoc_api_settings()
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{PAPER_TOPIC}}", topic)
        user_prompt = user_prompt.replace("{{WEEKLY_NOTES}}", weekly_content)
        
        # Estimate token count and wait for rate limit if necessary (disabled by default for postdoc bot)
        estimated_input_tokens = estimate_token_count(system_prompt + user_prompt)
        wait_for_rate_limit(estimated_input_tokens, tokens_per_minute_limit=25000, enabled=False)
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=api_settings['max_tokens'],
                temperature=api_settings['temperature'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            rating_text = response.content[0].text.strip()
            
            # Parse the rating as an integer
            try:
                rating = int(rating_text)
                if not 0 <= rating <= 100:
                    console.print(f"[yellow]Warning: Rating {rating} out of range, clamping to 0-100[/yellow]")
                    rating = max(0, min(100, rating))
                
                console.print(f"[green]✓ Rating: {rating}/100[/green]")
                return rating
            except ValueError:
                console.print(f"[red]Error: Could not parse rating from response: {rating_text}[/red]")
                raise ValueError(f"Invalid rating response: {rating_text}")
            
        except Exception as e:
            raise Exception(f"Failed to rate weekly note with Claude: {e}")
    
    def rate_all_notes(self, notes_folder: str, topic: str, paper_folder: Path = None) -> dict:
        """Rate all grad bot notes in a folder and save results to JSON in paper folder
        
        Args:
            notes_folder: Path to folder containing grad bot notes
            topic: Paper topic for rating relevance
            paper_folder: Path to paper folder for saving ratings (if None, saves to notes folder)
        
        Returns:
            Dictionary containing ratings data
        """
        notes_path = Path(notes_folder)
        if not notes_path.exists():
            raise FileNotFoundError(f"Notes folder not found: {notes_folder}")
        
        # Find all markdown files in the notes folder
        note_files = sorted([f for f in notes_path.glob("*.md")])
        
        if not note_files:
            raise FileNotFoundError(f"No markdown note files found in: {notes_folder}")
        
        console.print(f"[blue]Found {len(note_files)} note files to rate[/blue]")
        
        ratings = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for i, note_file in enumerate(note_files, 1):
                task = progress.add_task(f"Rating {i}/{len(note_files)}: {note_file.name}", total=None)
                
                try:
                    rating = self.rate_weekly_note(note_file, topic)
                    
                    # Extract week identifier from filename
                    week_identifier = note_file.stem  # e.g., "2025-07-20"
                    
                    ratings.append({
                        'file': note_file.name,
                        'week': week_identifier,
                        'rating': rating
                    })
                    
                    progress.update(task, description=f"✓ Rated: {note_file.name} ({rating}/100)")
                    
                except Exception as e:
                    console.print(f"[red]Error rating {note_file.name}: {e}[/red]")
                    ratings.append({
                        'file': note_file.name,
                        'week': note_file.stem,
                        'rating': None,
                        'error': str(e)
                    })
                    
                    progress.update(task, description=f"✗ Failed: {note_file.name}")
        
        # Create output data structure
        model = self.load_postdoc_model_config()
        output_data = {
            "metadata": {
                "topic": topic,
                "notes_folder": str(notes_folder),
                "rated_at": datetime.now().isoformat(),
                "model": model,
                "total_weeks": len(note_files),
                "successfully_rated": sum(1 for r in ratings if r.get('rating') is not None)
            },
            "ratings": ratings
        }
        
        # Save to JSON file in paper folder (or notes folder if no paper folder)
        output_filename = "postdoc_ratings.json"
        if paper_folder:
            output_path = paper_folder / output_filename
        else:
            output_path = notes_path / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓ Ratings saved to: {output_path}[/green]")
        
        # Show summary statistics
        valid_ratings = [r['rating'] for r in ratings if r.get('rating') is not None]
        if valid_ratings:
            avg_rating = sum(valid_ratings) / len(valid_ratings)
            max_rating = max(valid_ratings)
            min_rating = min(valid_ratings)
            
            console.print(f"\n[blue]Rating Summary:[/blue]")
            console.print(f"[blue]  Average: {avg_rating:.1f}/100[/blue]")
            console.print(f"[blue]  Range: {min_rating}-{max_rating}[/blue]")
            console.print(f"[blue]  Total weeks rated: {len(valid_ratings)}/{len(note_files)}[/blue]")
        
        return output_data


class ResearchAssistantBot:
    """Research assistant bot for identifying most relevant episodes for a paper topic"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def load_research_assistant_prompts(self) -> tuple[str, str]:
        """Load research assistant system and user prompts from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'research_assistant' not in config:
            raise ValueError("Research assistant prompt configuration not found in prompts.toml")
        
        prompt_config = config['research_assistant']
        system_prompt = prompt_config["system_prompt"]
        user_prompt = prompt_config["user_prompt"]
        
        return system_prompt, user_prompt
    
    def load_research_assistant_model_config(self) -> str:
        """Load research assistant model from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'models' not in config or 'research_assistant' not in config['models']:
            # Fallback to default model if not configured
            return "claude-sonnet-4-20250514"
        
        return config['models']['research_assistant']
    
    def load_research_assistant_api_settings(self) -> dict:
        """Load research assistant API settings from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        
        return {
            'temperature': config['api_settings']['research_assistant_temperature'],
            'max_tokens': config['api_settings']['research_assistant_max_tokens']
        }
    
    def load_grad_notes(self, notes_folder: str) -> str:
        """Load and combine all grad bot notes from a folder"""
        notes_path = Path(notes_folder)
        if not notes_path.exists():
            raise FileNotFoundError(f"Notes folder not found: {notes_folder}")
        
        # Find all markdown files in the notes folder
        note_files = sorted([f for f in notes_path.glob("*.md")])
        
        if not note_files:
            raise FileNotFoundError(f"No markdown note files found in: {notes_folder}")
        
        console.print(f"[blue]Loading {len(note_files)} grad bot note files[/blue]")
        
        # Combine all the grad bot notes
        combined_content = []
        combined_content.append("=== GRAD BOT ANALYSIS NOTES ===\n")
        
        for note_file in note_files:
            with open(note_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Extract just the content after the metadata header
            if "---" in content:
                # Split on the second occurrence of "---" to skip the metadata
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    main_content = parts[2].strip()
                else:
                    main_content = content
            else:
                main_content = content
            
            combined_content.append(f"\n=== WEEK FILE: {note_file.name} ===\n")
            combined_content.append(main_content)
            combined_content.append("\n" + "="*50 + "\n")
        
        combined_content.append("\n=== END GRAD NOTES ===")
        return "\n".join(combined_content)
    
    def find_script_file(self, episode_identifier: str, scripts_dir: Path) -> Optional[Path]:
        """Find the script file that matches the episode identifier"""
        # Extract season and episode number from identifier like "2x22 Becoming Part 2"
        match = re.match(r'^(\d+)x(\d+)', episode_identifier)
        if not match:
            return None
        
        season = match.group(1)
        episode = match.group(2)
        episode_prefix = f"{season}x{episode.zfill(2)}"
        
        # Look for files that start with the episode prefix
        for script_file in scripts_dir.glob(f"{episode_prefix} *.txt"):
            return script_file
        
        return None
    
    def copy_relevant_scripts(self, episode_list: List[str], paper_folder: Path, max_scripts: int = 5) -> List[str]:
        """Copy the top N relevant script files to a scripts subfolder in the paper folder
        
        Args:
            episode_list: List of episode identifiers ranked by relevance
            paper_folder: Path to the paper folder
            max_scripts: Maximum number of scripts to copy (default: 5)
        
        Returns:
            List of copied script filenames
        """
        # Limit to top N episodes
        top_episodes = episode_list[:max_scripts]
        
        scripts_dir = Path("scripts")
        if not scripts_dir.exists():
            raise FileNotFoundError("Scripts directory not found")
        
        scripts_output_dir = paper_folder / "scripts"
        scripts_output_dir.mkdir(exist_ok=True)
        
        copied_files = []
        
        console.print(f"[blue]Copying top {len(top_episodes)} relevant script files...[/blue]")
        
        for i, episode in enumerate(top_episodes, 1):
            script_file = self.find_script_file(episode, scripts_dir)
            
            if script_file:
                destination = scripts_output_dir / script_file.name
                shutil.copy2(script_file, destination)
                copied_files.append(script_file.name)
                console.print(f"[green]{i:2d}. Copied: {script_file.name}[/green]")
            else:
                console.print(f"[yellow]{i:2d}. Not found: {episode}[/yellow]")
        
        return copied_files
    
    def select_relevant_episodes_and_copy_scripts(self, notes_folder: str, paper_topic: str, topic_shorthand: str = None, target_paper_folder: Path = None, max_scripts: int = 5) -> tuple[List[str], List[str], Path]:
        """Analyze grad notes, select episodes, and copy relevant script files to papers folder
        
        Args:
            notes_folder: Path to folder containing grad bot notes
            paper_topic: Paper topic/thesis
            topic_shorthand: Short identifier for the topic
            target_paper_folder: Optional existing paper folder to use
            max_scripts: Maximum number of scripts to copy (default: 5)
        
        Returns:
            Tuple of (episode_list, copied_files, paper_folder)
        """
        
        console.print(f"[blue]Loading grad bot notes from: {notes_folder}[/blue]")
        combined_notes = self.load_grad_notes(notes_folder)
        
        console.print(f"[blue]Loading research assistant prompts[/blue]")
        system_prompt, user_prompt_template = self.load_research_assistant_prompts()
        
        console.print(f"[blue]Loading model configuration[/blue]")
        model = self.load_research_assistant_model_config()
        api_settings = self.load_research_assistant_api_settings()
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{PAPER_TOPIC}}", paper_topic)
        user_prompt = user_prompt.replace("{{GRAD_NOTES}}", combined_notes)
        
        console.print(f"[yellow]Analyzing episodes for topic: {paper_topic}[/yellow]")
        
        # Estimate token count and wait for rate limit if necessary (disabled by default for research assistant)
        estimated_input_tokens = estimate_token_count(system_prompt + user_prompt)
        wait_for_rate_limit(estimated_input_tokens, tokens_per_minute_limit=25000, enabled=False)
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=api_settings['max_tokens'],
                temperature=api_settings['temperature'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.content[0].text.strip()
            console.print(f"[green]✓ Episode selection complete[/green]")
            
            # Parse the JSON response
            try:
                import json
                
                # Try to extract JSON array from the response
                # Sometimes the LLM adds text before/after the JSON
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    episode_list = json.loads(json_text)
                else:
                    # No brackets found, try parsing the whole thing
                    episode_list = json.loads(response_text)
                
                if isinstance(episode_list, list):
                    # Use provided paper folder or create new one
                    if target_paper_folder is None:
                        paper_folder = create_paper_folder(topic_shorthand, notes_folder)
                        console.print(f"[blue]Created paper folder: {paper_folder}[/blue]")
                    else:
                        paper_folder = target_paper_folder
                        console.print(f"[blue]Using provided paper folder: {paper_folder}[/blue]")
                    
                    # Copy the top N relevant script files to the paper folder
                    copied_files = self.copy_relevant_scripts(episode_list, paper_folder, max_scripts)
                    return episode_list, copied_files, paper_folder
                else:
                    raise ValueError("Response is not a JSON list")
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing JSON response: {e}[/red]")
                console.print(f"[red]Raw response: {response_text}[/red]")
                raise Exception(f"Failed to parse episode selection response as JSON: {e}")
            
        except Exception as e:
            raise Exception(f"Failed to select episodes with Claude: {e}")


class ProfessorBot:
    """Professor bot for generating academic papers from curated research materials"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def load_professor_prompts(self) -> tuple[str, str]:
        """Load professor system and user prompts from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        prompt_type = "buffy"
        if prompt_type not in config:
            available = list(config.keys())
            raise ValueError(f"Prompt type '{prompt_type}' not found. Available: {available}")
        
        prompt_config = config[prompt_type]
        system_prompt = prompt_config["system_prompt"]
        user_prompt = prompt_config["user_prompt"]
        
        return system_prompt, user_prompt
    
    def load_professor_model_config(self) -> str:
        """Load professor model from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'models' not in config or 'paper_generation' not in config['models']:
            # Fallback to default model if not configured
            return "claude-sonnet-4-20250514"
        
        return config['models']['paper_generation']
    
    def load_professor_api_settings(self) -> dict:
        """Load professor API settings from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        
        return {
            'temperature': config['api_settings']['paper_generation_temperature'],
            'max_tokens': config['api_settings']['paper_generation_max_tokens']
        }
    
    def load_grad_notes(self, notes_folder: str, min_rating: int = 30, paper_folder: Path = None, 
                        verbatim_chat_threshold: int = 70, weekly_conversations_folder: str = None) -> str:
        """Load and combine grad bot notes from a folder, filtered by postdoc ratings
        
        Args:
            notes_folder: Path to folder containing grad bot notes
            min_rating: Minimum postdoc rating (0-100) for including a week's notes
            paper_folder: Path to paper folder containing ratings (if None, looks in notes folder)
            verbatim_chat_threshold: Rating threshold for using verbatim transcripts (default: 70)
            weekly_conversations_folder: Path to weekly conversation folder for verbatim transcripts
        
        Returns:
            Combined content from all notes meeting the rating threshold
        """
        notes_path = Path(notes_folder)
        if not notes_path.exists():
            raise FileNotFoundError(f"Notes folder not found: {notes_folder}")
        
        # Find all markdown files in the notes folder
        note_files = sorted([f for f in notes_path.glob("*.md")])
        
        if not note_files:
            raise FileNotFoundError(f"No markdown note files found in: {notes_folder}")
        
        # Load postdoc ratings - check paper folder first, then notes folder
        ratings_file = None
        if paper_folder and (paper_folder / "postdoc_ratings.json").exists():
            ratings_file = paper_folder / "postdoc_ratings.json"
        elif (notes_path / "postdoc_ratings.json").exists():
            ratings_file = notes_path / "postdoc_ratings.json"
        
        ratings_map = {}
        
        if ratings_file:
            console.print(f"[blue]Loading postdoc ratings from: {ratings_file}[/blue]")
            with open(ratings_file, 'r', encoding='utf-8') as f:
                ratings_data = json.load(f)
            
            # Create a map from filename to rating
            for rating_entry in ratings_data.get('ratings', []):
                if rating_entry.get('rating') is not None:
                    ratings_map[rating_entry['file']] = rating_entry['rating']
            
            console.print(f"[blue]Filtering notes with minimum rating: {min_rating}/100[/blue]")
            console.print(f"[blue]Using verbatim transcripts for ratings >= {verbatim_chat_threshold}/100[/blue]")
        else:
            console.print(f"[yellow]Warning: No postdoc_ratings.json found[/yellow]")
            console.print(f"[yellow]All notes will be included. Postdoc bot will run automatically during paper generation.[/yellow]")
        
        # Filter and combine notes
        combined_content = []
        combined_content.append("=== FILTERED CONVERSATION NOTES ===\n")
        
        included_count = 0
        excluded_count = 0
        verbatim_count = 0
        
        for note_file in note_files:
            # Check rating if available
            rating = None
            if ratings_map:
                rating = ratings_map.get(note_file.name)
                if rating is None:
                    console.print(f"[yellow]No rating found for {note_file.name}, excluding[/yellow]")
                    excluded_count += 1
                    continue
                elif rating < min_rating:
                    console.print(f"[dim]Excluding {note_file.name} (rating: {rating}/100)[/dim]")
                    excluded_count += 1
                    continue
            
            # Determine if we should use verbatim transcript or summarized notes
            use_verbatim = rating and rating >= verbatim_chat_threshold and weekly_conversations_folder
            
            if use_verbatim:
                # Load verbatim transcript from weekly conversations folder
                week_identifier = note_file.stem  # e.g., "2025-07-20"
                weekly_folder = Path(weekly_conversations_folder)
                
                # Find the matching weekly conversation file
                verbatim_file = None
                for conv_file in weekly_folder.glob(f"*_week_{week_identifier}.txt"):
                    verbatim_file = conv_file
                    break
                
                if verbatim_file and verbatim_file.exists():
                    console.print(f"[cyan]Including {note_file.name} as VERBATIM TRANSCRIPT (rating: {rating}/100)[/cyan]")
                    
                    with open(verbatim_file, 'r', encoding='utf-8') as f:
                        verbatim_content = f.read().strip()
                    
                    # Remove the header section if present
                    if "=== WEEK OF" in verbatim_content:
                        # Split at the first double newline after header
                        parts = verbatim_content.split('\n\n', 3)
                        if len(parts) >= 4:
                            verbatim_content = '\n\n'.join(parts[3:])
                    
                    combined_content.append(f"\n=== WEEK: {note_file.stem} (VERBATIM TRANSCRIPT) ===\n")
                    combined_content.append(verbatim_content)
                    combined_content.append("\n" + "="*50 + "\n")
                    included_count += 1
                    verbatim_count += 1
                else:
                    # Fall back to summarized notes if verbatim not found
                    console.print(f"[yellow]Verbatim transcript not found for {note_file.name}, using summarized notes (rating: {rating}/100)[/yellow]")
                    use_verbatim = False
            
            if not use_verbatim:
                # Use summarized grad bot notes
                console.print(f"[green]Including {note_file.name} (rating: {rating if rating else 'N/A'}/100)[/green]")
                
                # Load and process the note
                with open(note_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Extract just the content after the metadata header
                if "---" in content:
                    # Split on the second occurrence of "---" to skip the metadata
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        main_content = parts[2].strip()
                    else:
                        main_content = content
                else:
                    main_content = content
                
                combined_content.append(f"\n=== WEEK: {note_file.stem} ===\n")
                combined_content.append(main_content)
                combined_content.append("\n" + "="*50 + "\n")
                included_count += 1
        
        combined_content.append("\n=== END FILTERED NOTES ===")
        
        console.print(f"\n[blue]Notes summary:[/blue]")
        console.print(f"[green]  Included: {included_count} weeks[/green]")
        if verbatim_count > 0:
            console.print(f"[cyan]  Verbatim transcripts: {verbatim_count} weeks (rating >= {verbatim_chat_threshold})[/cyan]")
            console.print(f"[green]  Summarized notes: {included_count - verbatim_count} weeks[/green]")
        if excluded_count > 0:
            console.print(f"[dim]  Excluded: {excluded_count} weeks (below threshold)[/dim]")
        
        if included_count == 0:
            raise ValueError(f"No notes meet the minimum rating threshold of {min_rating}/100. Try lowering the threshold or running postdoc-bot first.")
        
        return "\n".join(combined_content)
    
    def load_episode_scripts(self, paper_folder: Path) -> str:
        """Load episode scripts from the paper folder"""
        scripts_dir = paper_folder / "scripts"
        if not scripts_dir.exists():
            console.print(f"[yellow]No scripts folder found in {paper_folder}[/yellow]")
            return ""
        
        script_files = sorted([f for f in scripts_dir.glob("*.txt")])
        if not script_files:
            console.print(f"[yellow]No script files found in {scripts_dir}[/yellow]")
            return ""
        
        console.print(f"[blue]Loading {len(script_files)} episode scripts[/blue]")
        
        # Combine all episode scripts
        combined_scripts = []
        combined_scripts.append("=== EPISODE SCRIPTS ===\n")
        
        for script_file in script_files:
            with open(script_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            episode_name = script_file.stem  # filename without extension
            combined_scripts.append(f"\n=== EPISODE: {episode_name} ===\n")
            combined_scripts.append(content)
            combined_scripts.append("\n" + "="*50 + "\n")
        
        combined_scripts.append("\n=== END SCRIPTS ===")
        return "\n".join(combined_scripts)
    
    def generate_paper(self, notes_folder: str, paper_folder: Path, topic: str, min_rating: int = 30,
                       verbatim_chat_threshold: int = 70, weekly_conversations_folder: str = None) -> str:
        """Generate a paper using Claude based on grad notes and episode scripts
        
        Args:
            notes_folder: Path to folder containing grad bot notes
            paper_folder: Path to paper folder containing scripts
            topic: Paper topic/thesis
            min_rating: Minimum postdoc rating for including notes (default: 30)
            verbatim_chat_threshold: Rating threshold for using verbatim transcripts (default: 70)
            weekly_conversations_folder: Path to weekly conversation folder for verbatim transcripts
        """
        
        console.print(f"[blue]Loading grad bot notes from: {notes_folder}[/blue]")
        grad_notes = self.load_grad_notes(notes_folder, min_rating, paper_folder, 
                                          verbatim_chat_threshold, weekly_conversations_folder)
        
        console.print(f"[blue]Loading episode scripts from: {paper_folder}[/blue]")
        episode_scripts = self.load_episode_scripts(paper_folder)
        
        console.print(f"[blue]Loading professor prompts[/blue]")
        system_prompt, user_prompt_template = self.load_professor_prompts()
        
        console.print(f"[blue]Loading model configuration[/blue]")
        model = self.load_professor_model_config()
        api_settings = self.load_professor_api_settings()
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{CHAT_TRANSCRIPT}}", grad_notes)
        user_prompt = user_prompt.replace("{{SCRIPT_TRANSCRIPTS}}", episode_scripts)
        user_prompt = user_prompt.replace("{{PAPER_TOPIC}}", topic)
        
        # Estimate token count for rate limiting
        estimated_input_tokens = estimate_token_count(system_prompt + user_prompt)
        console.print(f"[blue]Estimated input tokens: ~{estimated_input_tokens:,}[/blue]")
        
        # Wait for rate limit if necessary (25k tokens/minute)
        wait_for_rate_limit(estimated_input_tokens, tokens_per_minute_limit=25000)
        
        console.print(f"[yellow]Generating paper using model: {model}[/yellow]")
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=api_settings['max_tokens'],
                temperature=api_settings['temperature'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            paper_content = response.content[0].text
            console.print(f"[green]✓ Paper generated ({len(paper_content)} characters)[/green]")
            return paper_content
            
        except Exception as e:
            raise Exception(f"Failed to generate paper with Claude: {e}")
    
    def save_paper(self, content: str, topic: str, paper_folder: Path, notes_folder: str, model: str = None, min_rating: int = 30) -> Path:
        """Save generated paper to the paper folder"""
        # Simple filename - just "paper.md" since we're in a topic-specific folder
        filename = "paper.md"
        output_path = paper_folder / filename
        
        # Get model for metadata if not provided
        if model is None:
            model = self.load_professor_model_config()
        
        metadata = f"""---
title: "{topic}"
source_notes: "{notes_folder}"
generated_at: "{datetime.now().isoformat()}"
model: "{model}"
generation_method: "professor_bot"
min_rating_threshold: {min_rating}
---

"""
        
        full_content = metadata + content
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return output_path


class ReviewerBot:
    """Peer reviewer bot for evaluating academic papers on Buffy Studies"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def load_reviewer_prompts(self) -> tuple[str, str]:
        """Load reviewer system and user prompts from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'reviewer_bot' not in config:
            raise ValueError("Reviewer bot prompt configuration not found in prompts.toml")
        
        prompt_config = config['reviewer_bot']
        system_prompt = prompt_config["system_prompt"]
        user_prompt = prompt_config["user_prompt"]
        
        return system_prompt, user_prompt
    
    def load_reviewer_model_config(self) -> str:
        """Load reviewer model from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'models' not in config or 'paper_generation' not in config['models']:
            # Use paper_generation model as reviewer needs similar capabilities
            return "claude-sonnet-4-20250514"
        
        return config['models']['paper_generation']
    
    def load_reviewer_api_settings(self) -> dict:
        """Load reviewer API settings from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        
        return {
            'temperature': config['api_settings']['paper_generation_temperature'],
            'max_tokens': config['api_settings']['paper_generation_max_tokens']
        }
    
    def load_paper(self, paper_folder: str) -> str:
        """Load the paper content from paper.md"""
        paper_path = Path(paper_folder)
        if not paper_path.exists():
            raise FileNotFoundError(f"Paper folder not found: {paper_folder}")
        
        paper_file = paper_path / "paper.md"
        if not paper_file.exists():
            raise FileNotFoundError(f"paper.md not found in {paper_folder}")
        
        console.print(f"[blue]Loading paper from: {paper_file}[/blue]")
        
        with open(paper_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        return content
    
    def load_scripts(self, paper_folder: str) -> str:
        """Load episode scripts from the paper folder"""
        paper_path = Path(paper_folder)
        scripts_dir = paper_path / "scripts"
        
        if not scripts_dir.exists():
            console.print(f"[yellow]No scripts folder found in {paper_folder}[/yellow]")
            return "No episode scripts provided."
        
        script_files = sorted([f for f in scripts_dir.glob("*.txt")])
        if not script_files:
            console.print(f"[yellow]No script files found in {scripts_dir}[/yellow]")
            return "No episode scripts provided."
        
        console.print(f"[blue]Loading {len(script_files)} episode scripts[/blue]")
        
        # Combine all episode scripts
        combined_scripts = []
        combined_scripts.append("=== EPISODE SCRIPTS ===\n")
        
        for script_file in script_files:
            with open(script_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            episode_name = script_file.stem  # filename without extension
            combined_scripts.append(f"\n=== EPISODE: {episode_name} ===\n")
            combined_scripts.append(content)
            combined_scripts.append("\n" + "="*50 + "\n")
        
        combined_scripts.append("\n=== END SCRIPTS ===")
        return "\n".join(combined_scripts)
    
    def review_paper(self, paper_folder: str) -> dict:
        """Review a paper and return the review data
        
        Args:
            paper_folder: Path to folder containing paper.md and scripts/
        
        Returns:
            Dictionary containing the review
        """
        console.print(f"[blue]Loading paper from: {paper_folder}[/blue]")
        paper_content = self.load_paper(paper_folder)
        
        console.print(f"[blue]Loading episode scripts[/blue]")
        episode_scripts = self.load_scripts(paper_folder)
        
        console.print(f"[blue]Loading reviewer prompts[/blue]")
        system_prompt, user_prompt_template = self.load_reviewer_prompts()
        
        console.print(f"[blue]Loading model configuration[/blue]")
        model = self.load_reviewer_model_config()
        api_settings = self.load_reviewer_api_settings()
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{PAPER_CONTENT}}", paper_content)
        user_prompt = user_prompt.replace("{{EPISODE_SCRIPTS}}", episode_scripts)
        
        # Estimate token count for rate limiting (enabled for reviewer bot since it processes full papers)
        estimated_input_tokens = estimate_token_count(system_prompt + user_prompt)
        console.print(f"[blue]Estimated input tokens: ~{estimated_input_tokens:,}[/blue]")
        
        # Wait for rate limit if necessary (25k tokens/minute)
        wait_for_rate_limit(estimated_input_tokens, tokens_per_minute_limit=25000, enabled=True)
        
        console.print(f"[yellow]Reviewing paper using model: {model}[/yellow]")
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=api_settings['max_tokens'],
                temperature=api_settings['temperature'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.content[0].text.strip()
            console.print(f"[green]✓ Review complete[/green]")
            
            # Parse the JSON response
            try:
                # Try to extract JSON object from the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    review_data = json.loads(json_text)
                else:
                    # No braces found, try parsing the whole thing
                    review_data = json.loads(response_text)
                
                if not isinstance(review_data, dict) or 'decision' not in review_data:
                    raise ValueError("Response does not contain 'decision' key")
                
                return review_data
                
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing JSON response: {e}[/red]")
                console.print(f"[red]Raw response: {response_text[:500]}...[/red]")
                raise Exception(f"Failed to parse review response as JSON: {e}")
            
        except Exception as e:
            raise Exception(f"Failed to review paper with Claude: {e}")
    
    def save_review(self, review_data: dict, paper_folder: str) -> Path:
        """Save review to a JSON file in the paper folder's reviews/ subdirectory
        
        Args:
            review_data: Dictionary containing review
            paper_folder: Path to paper folder
        
        Returns:
            Path to saved review file
        """
        paper_path = Path(paper_folder)
        
        # Create reviews subdirectory
        reviews_dir = paper_path / "reviews"
        reviews_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = reviews_dir / f"review_paper_{timestamp}.json"
        
        # Prepare output data with metadata
        model = self.load_reviewer_model_config()
        output_data = {
            "metadata": {
                "paper_folder": str(paper_folder),
                "reviewed_at": datetime.now().isoformat(),
                "model": model,
                "decision": review_data.get('decision', 'UNKNOWN')
            },
            "review": review_data
        }
        
        # Save to JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ Review saved to: {json_path}[/green]")
        
        return json_path


class ResearcherBot:
    """Researcher bot for generating paper abstracts/proposals based on grad notes"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def load_researcher_prompts(self) -> tuple[str, str]:
        """Load researcher system and user prompts from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'researcher_bot' not in config:
            raise ValueError("Researcher bot prompt configuration not found in prompts.toml")
        
        prompt_config = config['researcher_bot']
        system_prompt = prompt_config["system_prompt"]
        user_prompt = prompt_config["user_prompt"]
        
        return system_prompt, user_prompt
    
    def load_researcher_model_config(self) -> str:
        """Load researcher model from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if 'models' not in config or 'paper_generation' not in config['models']:
            # Use paper_generation model as it's similar in scope to professor bot
            return "claude-sonnet-4-20250514"
        
        return config['models']['paper_generation']
    
    def load_researcher_api_settings(self) -> dict:
        """Load researcher API settings from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        
        return {
            'temperature': config['api_settings']['paper_generation_temperature'],
            'max_tokens': config['api_settings']['paper_generation_max_tokens']
        }
    
    def load_grad_notes(self, notes_folder: str) -> str:
        """Load and combine all grad bot notes from a folder"""
        notes_path = Path(notes_folder)
        if not notes_path.exists():
            raise FileNotFoundError(f"Notes folder not found: {notes_folder}")
        
        # Find all markdown files in the notes folder
        note_files = sorted([f for f in notes_path.glob("*.md")])
        
        if not note_files:
            raise FileNotFoundError(f"No markdown note files found in: {notes_folder}")
        
        console.print(f"[blue]Loading {len(note_files)} grad bot note files[/blue]")
        
        # Combine all the grad bot notes
        combined_content = []
        combined_content.append("=== GRADUATE RESEARCH NOTES ===\n")
        
        for note_file in note_files:
            with open(note_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Extract just the content after the metadata header
            if "---" in content:
                # Split on the second occurrence of "---" to skip the metadata
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    main_content = parts[2].strip()
                else:
                    main_content = content
            else:
                main_content = content
            
            combined_content.append(f"\n=== WEEK: {note_file.stem} ===\n")
            combined_content.append(main_content)
            combined_content.append("\n" + "="*50 + "\n")
        
        combined_content.append("\n=== END GRADUATE NOTES ===")
        return "\n".join(combined_content)
    
    def generate_abstracts(self, notes_folder: str) -> dict:
        """Generate paper abstracts based on grad notes
        
        Args:
            notes_folder: Path to folder containing grad bot notes
        
        Returns:
            Dictionary containing the abstracts and metadata
        """
        console.print(f"[blue]Loading grad bot notes from: {notes_folder}[/blue]")
        grad_notes = self.load_grad_notes(notes_folder)
        
        console.print(f"[blue]Loading researcher prompts[/blue]")
        system_prompt, user_prompt_template = self.load_researcher_prompts()
        
        console.print(f"[blue]Loading model configuration[/blue]")
        model = self.load_researcher_model_config()
        api_settings = self.load_researcher_api_settings()
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{GRAD_NOTES}}", grad_notes)
        
        # Estimate token count for rate limiting (enabled for researcher bot)
        estimated_input_tokens = estimate_token_count(system_prompt + user_prompt)
        console.print(f"[blue]Estimated input tokens: ~{estimated_input_tokens:,}[/blue]")
        
        # Wait for rate limit if necessary (25k tokens/minute)
        wait_for_rate_limit(estimated_input_tokens, tokens_per_minute_limit=25000, enabled=True)
        
        console.print(f"[yellow]Generating paper abstracts using model: {model}[/yellow]")
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=api_settings['max_tokens'],
                temperature=api_settings['temperature'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.content[0].text.strip()
            console.print(f"[green]✓ Abstracts generated[/green]")
            
            # Parse the JSON response
            try:
                # Try to extract JSON object from the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    abstracts_data = json.loads(json_text)
                else:
                    # No braces found, try parsing the whole thing
                    abstracts_data = json.loads(response_text)
                
                if not isinstance(abstracts_data, dict) or 'abstracts' not in abstracts_data:
                    raise ValueError("Response does not contain 'abstracts' key")
                
                return abstracts_data
                
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing JSON response: {e}[/red]")
                console.print(f"[red]Raw response: {response_text[:500]}...[/red]")
                raise Exception(f"Failed to parse abstracts response as JSON: {e}")
            
        except Exception as e:
            raise Exception(f"Failed to generate abstracts with Claude: {e}")
    
    def save_abstracts(self, abstracts_data: dict, notes_folder: str) -> Path:
        """Save generated abstracts to a JSON file in the grad notes folder
        
        Args:
            abstracts_data: Dictionary containing abstracts
            notes_folder: Path to grad notes folder
        
        Returns:
            Path to saved file
        """
        notes_path = Path(notes_folder)
        
        # Add metadata
        model = self.load_researcher_model_config()
        output_data = {
            "metadata": {
                "notes_folder": str(notes_folder),
                "generated_at": datetime.now().isoformat(),
                "model": model,
                "total_abstracts": len(abstracts_data.get('abstracts', []))
            },
            "abstracts": abstracts_data.get('abstracts', [])
        }
        
        # Save to JSON file in the grad notes folder
        output_filename = "paper_abstracts.json"
        output_path = notes_path / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ Abstracts saved to: {output_path}[/green]")
        
        # Show summary statistics
        abstracts = abstracts_data.get('abstracts', [])
        if abstracts:
            avg_rating = sum(a.get('rating', 0) for a in abstracts) / len(abstracts)
            max_rating = max(a.get('rating', 0) for a in abstracts)
            
            console.print(f"\n[blue]Abstracts Summary:[/blue]")
            console.print(f"[blue]  Total abstracts generated: {len(abstracts)}[/blue]")
            console.print(f"[blue]  Average rating: {avg_rating:.1f}/100[/blue]")
            console.print(f"[blue]  Highest rating: {max_rating}/100[/blue]")
            
            # Show top 3 abstracts
            sorted_abstracts = sorted(abstracts, key=lambda x: x.get('rating', 0), reverse=True)
            console.print(f"\n[green]Top abstracts:[/green]")
            for i, abstract in enumerate(sorted_abstracts[:3], 1):
                console.print(f"[yellow]{i}. {abstract.get('title', 'Untitled')}[/yellow] (rating: {abstract.get('rating', 0)}/100)")
        
        return output_path


@click.group()
def cli():
    """Zulip conversation extractor"""
    pass


@cli.command()
@click.option('--stream', required=True, help='Stream name')
@click.option('--topic', required=True, help='Topic name')
@click.option('--limit', default=1000, help='Maximum number of messages to extract')
@click.option('--output', help='Output filename (default: stream_topic.json)')
def extract_stream(stream, topic, limit, output):
    """Extract messages from a stream topic"""
    api_key = os.getenv('ZULIP_API_KEY')
    site_url = os.getenv('ZULIP_SITE_URL')
    
    if not api_key or not site_url:
        console.print("[red]Error: ZULIP_API_KEY and ZULIP_SITE_URL must be set in environment variables[/red]")
        return
    
    email = os.getenv('ZULIP_EMAIL')
    extractor = ZulipExtractor(api_key, site_url, email)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Extracting messages...", total=None)
        
        try:
            messages = extractor.extract_stream_messages(stream, topic, limit)
            progress.update(task, description=f"Found {len(messages)} messages")
            
            if not output:
                output = f"{stream}_{topic}.json"
            
            extractor.save_messages(messages, output)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--users', help='Comma-separated list of user emails (if not provided, uses default from env)')
@click.option('--limit', default=1000, help='Maximum number of messages to extract (automatically batched for limits > 5000)')
@click.option('--output', help='Output filename (default: private_conversation.json)')
@click.option('--weekly-chunks', is_flag=True, help='Also split conversation into weekly chunks for multi-LLM processing')
def extract_private(users, limit, output, weekly_chunks):
    """Extract messages from a private conversation using email addresses"""
    api_key = os.getenv('ZULIP_API_KEY')
    site_url = os.getenv('ZULIP_SITE_URL')
    
    if not api_key or not site_url:
        console.print("[red]Error: ZULIP_API_KEY and ZULIP_SITE_URL must be set in environment variables[/red]")
        return
    
    # Load default users from environment variable if not provided
    if not users:
        users = os.getenv('ZULIP_RECIPIENT')
        if users:
            console.print(f"[blue]Using default recipient from ZULIP_RECIPIENT environment variable[/blue]")
        else:
            console.print("[red]Error: No --users provided and ZULIP_RECIPIENT environment variable not set[/red]")
            console.print("[yellow]Either provide --users or set ZULIP_RECIPIENT=email@example.com in your .env file[/yellow]")
            return
    
    user_emails = [email.strip() for email in users.split(',')]
    email = os.getenv('ZULIP_EMAIL')
    extractor = ZulipExtractor(api_key, site_url, email)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Extracting messages...", total=None)
        try:
            messages = extractor.extract_private_messages(user_emails, limit)
            progress.update(task, description=f"Found {len(messages)} messages")
            
            if not output:
                output = "private_conversation.json"
            
            extractor.save_messages(messages, output)
            
            # Split into weekly chunks if requested
            if weekly_chunks:
                console.print("\n[blue]Creating weekly chunks for multi-LLM processing...[/blue]")
                extractor.split_into_weekly_chunks(messages, output)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--conversation', required=True, help='Path to conversation JSON file')
@click.option('--topic', required=True, help='Paper topic/thesis')
@click.option('--output', help='Custom output filename (optional)')
@click.option('--topic-shorthand', help='Short identifier for the topic (used in filenames if --output not provided)')
def generate_paper_direct(conversation, topic, output, topic_shorthand):
    """Generate an academic paper directly from a conversation transcript (legacy direct approach)"""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        console.print("[red]Error: CLAUDE_API_KEY must be set in environment variables[/red]")
        console.print("[yellow]Add CLAUDE_API_KEY=your-api-key to your .env file[/yellow]")
        return
    
    generator = PaperGenerator(claude_api_key)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating paper...", total=None)
        
        try:
            # Generate the paper
            paper_content = generator.generate_paper(conversation, topic)
            progress.update(task, description="Saving paper...")
            
            # Get model for metadata
            model = generator.load_model_config()
            
            # Save the paper using the new folder structure
            if output:
                # If custom output provided, create a paper folder and use custom filename
                paper_folder = create_paper_folder(topic_shorthand)
                output_path = paper_folder / output
                if not output.endswith('.md'):
                    output_path = output_path.with_suffix('.md')
                
                # Add metadata header
                metadata = f"""---
title: "{topic}"
source_conversation: "{conversation}"
generated_at: "{datetime.now().isoformat()}"
model: "{model}"
---

"""
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(metadata + paper_content)
            else:
                # Use automatic filename with folder structure
                output_path = generator.save_paper(paper_content, topic, conversation, topic_shorthand, model)
            
            progress.update(task, description=f"Paper saved!")
            console.print(f"[green]✓ Paper saved to: {output_path}[/green]")
            
            # Show a preview of the paper
            console.print("\n[blue]Paper preview:[/blue]")
            preview = paper_content[:500] + "..." if len(paper_content) > 500 else paper_content
            console.print(Markdown(preview))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--weekly-dir', required=True, help='Path to directory containing weekly chunk files')
@click.option('--specific-week', help='Specific week to rerun (e.g., "2025-07-20"). If provided, only this week will be processed.')
@click.option('--output-folder', help='Existing grad_bot_analysis folder to save results to (e.g., "grad_notes/grad_bot_analysis_20251013_144539")')
def run_grad_bots(weekly_dir, specific_week, output_folder):
    """Run grad bot analysis on all weekly conversation chunks or a specific week"""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        console.print("[red]Error: CLAUDE_API_KEY must be set in environment variables[/red]")
        console.print("[yellow]Add CLAUDE_API_KEY=your-api-key to your .env file[/yellow]")
        return
    
    grad_bot = GradBot(claude_api_key)
    
    try:
        if specific_week:
            # Process only the specific week
            console.print(f"[blue]Starting grad bot analysis for specific week: {specific_week}[/blue]")
            console.print(f"[blue]Weekly directory: {weekly_dir}[/blue]")
            if output_folder:
                console.print(f"[blue]Output folder: {output_folder}[/blue]")
            
            results = grad_bot.process_specific_weekly_chunk(weekly_dir, specific_week, output_folder)
        else:
            # Process all weekly chunks (original behavior)
            console.print(f"[blue]Starting grad bot analysis for all weeks...[/blue]")
            console.print(f"[blue]Weekly directory: {weekly_dir}[/blue]")
            if output_folder:
                console.print(f"[yellow]Warning: --output-folder is ignored when processing all weeks (creates new folder)[/yellow]")
            
            results = grad_bot.process_all_weekly_chunks(weekly_dir)
        
        # Show summary
        if results:
            successful_files = [r['analysis_file'] for r in results if r['success']]
            
            if successful_files:
                console.print(f"\n[green]Grad bot notes created:[/green]")
                for analysis_file in successful_files[:5]:  # Show first 5
                    console.print(f"[blue]  - {analysis_file}[/blue]")
                if len(successful_files) > 5:
                    console.print(f"[blue]  ... and {len(successful_files) - 5} more[/blue]")
                
                console.print(f"\n[yellow]Next steps:[/yellow]")
                console.print(f"[yellow]1. Review the grad bot notes in the 'grad_notes/' directory[/yellow]")
                console.print(f"[yellow]2. Use the synthesized transcripts to generate your final paper[/yellow]")
            
            failed_count = sum(1 for r in results if not r['success'])
            if failed_count > 0:
                console.print(f"\n[red]Failed to process {failed_count} weeks[/red]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _generate_papers_from_abstracts(
    claude_api_key: str,
    notes_folder: str = None,
    topic_shorthand: str = None,
    min_rating: int = 30,
    max_scripts: int = 5,
    verbatim_chat_threshold: int = 70,
    weekly_conversations: str = 'conversations/weekly',
    min_abstract_rating: int = 0
):
    """Helper function to generate papers for all abstracts from researcher bot"""
    
    # Use most recent folder if not provided
    if not notes_folder:
        notes_folder = str(get_most_recent_grad_notes_folder())
    
    # Load abstracts from JSON file
    notes_path = Path(notes_folder)
    abstracts_file = notes_path / "paper_abstracts.json"
    
    if not abstracts_file.exists():
        console.print(f"[red]Error: paper_abstracts.json not found in {notes_folder}[/red]")
        console.print(f"[yellow]Run 'uv run main.py researcher-bot' first to generate abstracts[/yellow]")
        return
    
    console.print(f"[blue]Loading abstracts from: {abstracts_file}[/blue]")
    
    with open(abstracts_file, 'r', encoding='utf-8') as f:
        abstracts_data = json.load(f)
    
    abstracts = abstracts_data.get('abstracts', [])
    
    if not abstracts:
        console.print(f"[yellow]No abstracts found in {abstracts_file}[/yellow]")
        return
    
    # Filter abstracts by rating
    filtered_abstracts = [
        a for a in abstracts 
        if a.get('rating', 0) >= min_abstract_rating
    ]
    
    if not filtered_abstracts:
        console.print(f"[yellow]No abstracts meet the minimum rating threshold of {min_abstract_rating}/100[/yellow]")
        console.print(f"[blue]Total abstracts: {len(abstracts)}, Filtered: 0[/blue]")
        return
    
    console.print(f"\n[green]{'='*80}[/green]")
    console.print(f"[green]Generating Papers from Researcher Bot Abstracts[/green]")
    console.print(f"[green]{'='*80}[/green]\n")
    console.print(f"[blue]Total abstracts: {len(abstracts)}[/blue]")
    console.print(f"[blue]Abstracts meeting threshold (>={min_abstract_rating}): {len(filtered_abstracts)}[/blue]")
    console.print(f"[blue]Notes folder: {notes_folder}[/blue]\n")
    
    # Sort by rating (highest first)
    sorted_abstracts = sorted(filtered_abstracts, key=lambda x: x.get('rating', 0), reverse=True)
    
    successful_papers = []
    failed_papers = []
    
    for i, abstract_entry in enumerate(sorted_abstracts, 1):
        title = abstract_entry.get('title', f'Untitled Abstract {i}')
        abstract_text = abstract_entry.get('abstract', '')
        rating = abstract_entry.get('rating', 0)
        
        console.print(f"\n[cyan]{'='*80}[/cyan]")
        console.print(f"[cyan]Paper {i}/{len(sorted_abstracts)}: {title}[/cyan]")
        console.print(f"[cyan]Rating: {rating}/100[/cyan]")
        console.print(f"[cyan]{'='*80}[/cyan]\n")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                # Use most recent folder if not provided
                if not notes_folder:
                    notes_folder_resolved = str(get_most_recent_grad_notes_folder())
                else:
                    notes_folder_resolved = notes_folder
                
                # Create paper folder for this abstract
                # Use title as shorthand, sanitized
                title_shorthand = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                title_shorthand = title_shorthand.replace(' ', '_')[:50]  # Limit length
                
                target_paper_folder = create_paper_folder(title_shorthand, notes_folder_resolved)
                console.print(f"[blue]Created paper folder: {target_paper_folder}[/blue]")
                
                # Step 1: Run postdoc bot to rate notes (if not already done)
                ratings_file = target_paper_folder / "postdoc_ratings.json"
                if not ratings_file.exists():
                    task0 = progress.add_task("Running postdoc bot...", total=None)
                    console.print(f"[blue]Running postdoc bot to rate notes for relevance[/blue]")
                    
                    postdoc_bot_instance = PostdocBot(claude_api_key)
                    postdoc_bot_instance.rate_all_notes(notes_folder_resolved, abstract_text, target_paper_folder)
                    
                    progress.update(task0, description="✓ Postdoc bot complete")
                else:
                    console.print(f"[blue]Postdoc ratings already exist in {target_paper_folder}, skipping postdoc bot[/blue]")
                
                # Step 2: Run research assistant if needed (to get episodes and copy scripts)
                scripts_dir = target_paper_folder / "scripts"
                if not (scripts_dir.exists() and list(scripts_dir.glob("*.txt"))):
                    task1 = progress.add_task("Running research assistant...", total=None)
                    console.print(f"[blue]Running research assistant to find relevant episodes and copy scripts[/blue]")
                    console.print(f"[blue]Max scripts to copy: {max_scripts}[/blue]")
                    
                    research_assistant_bot = ResearchAssistantBot(claude_api_key)
                    episode_list, copied_files, _ = research_assistant_bot.select_relevant_episodes_and_copy_scripts(
                        notes_folder_resolved, abstract_text, title_shorthand, target_paper_folder, max_scripts
                    )
                    
                    console.print(f"[green]✓ Research assistant copied {len(copied_files)} script files[/green]")
                    progress.update(task1, description="✓ Research assistant complete")
                else:
                    console.print(f"[blue]Scripts already exist in {scripts_dir}, skipping research assistant[/blue]")
                
                # Step 3: Run professor bot to generate paper
                task2 = progress.add_task("Running professor bot...", total=None)
                
                professor_bot = ProfessorBot(claude_api_key)
                
                console.print(f"[blue]Running professor bot to generate paper[/blue]")
                paper_content = professor_bot.generate_paper(
                    notes_folder_resolved, target_paper_folder, abstract_text, min_rating,
                    verbatim_chat_threshold, weekly_conversations
                )
                
                progress.update(task2, description="Saving paper...")
                
                # Save the paper
                output_path = professor_bot.save_paper(paper_content, title, target_paper_folder, notes_folder_resolved, min_rating=min_rating)
                
                progress.update(task2, description="✓ Professor bot complete")
                
                console.print(f"[green]✓ Paper generated and saved to: {output_path}[/green]")
                
                successful_papers.append({
                    'title': title,
                    'rating': rating,
                    'path': str(output_path),
                    'folder': str(target_paper_folder)
                })
                
        except Exception as e:
            console.print(f"[red]Error generating paper for '{title}': {e}[/red]")
            failed_papers.append({
                'title': title,
                'rating': rating,
                'error': str(e)
            })
    
    # Final summary
    console.print(f"\n[green]{'='*80}[/green]")
    console.print(f"[green]Batch Paper Generation Complete[/green]")
    console.print(f"[green]{'='*80}[/green]\n")
    console.print(f"[blue]Successfully generated: {len(successful_papers)}/{len(sorted_abstracts)} papers[/blue]")
    
    if successful_papers:
        console.print(f"\n[green]Generated papers:[/green]")
        for paper in successful_papers:
            console.print(f"[blue]  • {paper['title']}[/blue] (rating: {paper['rating']}/100)")
            console.print(f"[dim]    {paper['folder']}[/dim]")
    
    if failed_papers:
        console.print(f"\n[red]Failed papers:[/red]")
        for paper in failed_papers:
            console.print(f"[red]  • {paper['title']}[/red] (rating: {paper['rating']}/100)")
            console.print(f"[dim]    Error: {paper['error']}[/dim]")


@cli.command()
@click.option('--notes-folder', help='Path to folder containing grad bot notes (e.g., grad_notes/nietzsche_20250107_120000). If not provided, uses most recent folder.')
@click.option('--topic', help='Paper topic/thesis (required unless using --from-abstracts)')
@click.option('--topic-shorthand', help='Short identifier for the topic (used in folder naming if no existing paper folder)')
@click.option('--paper-folder', help='Path to existing paper folder (if research assistant already created one)')
@click.option('--min-rating', default=30, type=int, help='Minimum postdoc rating (0-100) for including weekly notes in paper generation (default: 30)')
@click.option('--max-scripts', default=5, type=int, help='Maximum number of episode scripts to copy (default: 5)')
@click.option('--verbatim-chat-threshold', default=70, type=int, help='Rating threshold (0-100) for using verbatim transcripts instead of summarized notes (default: 70)')
@click.option('--weekly-conversations', default='conversations/weekly', help='Path to weekly conversation folder for verbatim transcripts (default: conversations/weekly)')
@click.option('--from-abstracts', is_flag=True, help='Generate papers for all abstracts from researcher bot (uses paper_abstracts.json)')
@click.option('--min-abstract-rating', default=0, type=int, help='Minimum abstract rating (0-100) for generating papers when using --from-abstracts (default: 0)')
def generate_paper(notes_folder, topic, topic_shorthand, paper_folder, min_rating, max_scripts, verbatim_chat_threshold, weekly_conversations, from_abstracts, min_abstract_rating):
    """Generate an academic paper by running postdoc bot, research assistant, and professor bot in sequence"""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        console.print("[red]Error: CLAUDE_API_KEY must be set in environment variables[/red]")
        console.print("[yellow]Add CLAUDE_API_KEY=your-api-key to your .env file[/yellow]")
        return
    
    # Validate that either topic or from_abstracts is provided
    if not from_abstracts and not topic:
        console.print("[red]Error: Either --topic or --from-abstracts must be provided[/red]")
        return
    
    if from_abstracts and topic:
        console.print("[yellow]Warning: Both --topic and --from-abstracts provided. Using --from-abstracts (--topic will be ignored)[/yellow]")
    
    # If using abstracts, delegate to batch generation function
    if from_abstracts:
        _generate_papers_from_abstracts(
            claude_api_key=claude_api_key,
            notes_folder=notes_folder,
            topic_shorthand=topic_shorthand,
            min_rating=min_rating,
            max_scripts=max_scripts,
            verbatim_chat_threshold=verbatim_chat_threshold,
            weekly_conversations=weekly_conversations,
            min_abstract_rating=min_abstract_rating
        )
        return
    
    # Validate min_rating
    if not 0 <= min_rating <= 100:
        console.print(f"[red]Error: --min-rating must be between 0 and 100, got {min_rating}[/red]")
        return
    
    # Validate verbatim_chat_threshold
    if not 0 <= verbatim_chat_threshold <= 100:
        console.print(f"[red]Error: --verbatim-chat-threshold must be between 0 and 100, got {verbatim_chat_threshold}[/red]")
        return
    
    # Validate max_scripts
    if max_scripts < 1:
        console.print(f"[red]Error: --max-scripts must be at least 1, got {max_scripts}[/red]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        try:
            # Use most recent folder if not provided
            if not notes_folder:
                notes_folder = str(get_most_recent_grad_notes_folder())
            # Determine paper folder - create early so both bots use the same one
            target_paper_folder = None
            if paper_folder:
                # Use provided paper folder
                target_paper_folder = Path(paper_folder)
                if not target_paper_folder.exists():
                    console.print(f"[red]Error: Paper folder not found: {paper_folder}[/red]")
                    return
                console.print(f"[blue]Using existing paper folder: {target_paper_folder}[/blue]")
            else:
                # Create new paper folder using shared function
                target_paper_folder = create_paper_folder(topic_shorthand, notes_folder)
                console.print(f"[blue]Created paper folder: {target_paper_folder}[/blue]")
            
            # Step 1: Run postdoc bot to rate notes (if not already done)
            ratings_file = target_paper_folder / "postdoc_ratings.json"
            if not ratings_file.exists():
                task0 = progress.add_task("Running postdoc bot...", total=None)
                console.print(f"[blue]Running postdoc bot to rate notes for relevance[/blue]")
                
                postdoc_bot_instance = PostdocBot(claude_api_key)
                postdoc_bot_instance.rate_all_notes(notes_folder, topic, target_paper_folder)
                
                progress.update(task0, description="✓ Postdoc bot complete")
            else:
                console.print(f"[blue]Postdoc ratings already exist in {target_paper_folder}, skipping postdoc bot[/blue]")
            
            # Step 2: Run research assistant if needed (to get episodes and copy scripts)
            scripts_dir = target_paper_folder / "scripts"
            if not (scripts_dir.exists() and list(scripts_dir.glob("*.txt"))):
                task1 = progress.add_task("Running research assistant...", total=None)
                console.print(f"[blue]Running research assistant to find relevant episodes and copy scripts[/blue]")
                console.print(f"[blue]Max scripts to copy: {max_scripts}[/blue]")
                
                research_assistant_bot = ResearchAssistantBot(claude_api_key)
                episode_list, copied_files, _ = research_assistant_bot.select_relevant_episodes_and_copy_scripts(
                    notes_folder, topic, topic_shorthand, target_paper_folder, max_scripts
                )
                
                console.print(f"[green]✓ Research assistant copied {len(copied_files)} script files[/green]")
                progress.update(task1, description="✓ Research assistant complete")
            else:
                console.print(f"[blue]Scripts already exist in {scripts_dir}, skipping research assistant[/blue]")
            
            # Step 3: Run professor bot to generate paper
            task2 = progress.add_task("Running professor bot...", total=None)
            
            professor_bot = ProfessorBot(claude_api_key)
            
            console.print(f"[blue]Running professor bot to generate paper[/blue]")
            paper_content = professor_bot.generate_paper(notes_folder, target_paper_folder, topic, min_rating,
                                                        verbatim_chat_threshold, weekly_conversations)
            
            progress.update(task2, description="Saving paper...")
            
            # Save the paper
            output_path = professor_bot.save_paper(paper_content, topic, target_paper_folder, notes_folder, min_rating=min_rating)
            
            progress.update(task2, description="✓ Professor bot complete")
            
            console.print(f"[green]✓ Paper generated and saved to: {output_path}[/green]")
            console.print(f"[blue]Paper folder contents: {target_paper_folder}[/blue]")
            
            # Show a preview of the paper
            console.print("\n[blue]Paper preview:[/blue]")
            preview = paper_content[:500] + "..." if len(paper_content) > 500 else paper_content
            console.print(Markdown(preview))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--notes-folder', help='Path to folder containing grad bot notes (e.g., grad_notes/nietzsche_20250107_120000). If not provided, uses most recent folder.')
@click.option('--topic', required=True, help='Paper topic/thesis to find relevant episodes for')
@click.option('--topic-shorthand', help='Short identifier for the topic (used in folder naming)')
@click.option('--output-format', default='list', type=click.Choice(['list', 'json']), help='Output format: list (human-readable) or json')
@click.option('--max-scripts', default=5, type=int, help='Maximum number of episode scripts to copy (default: 5)')
def research_assistant(notes_folder, topic, topic_shorthand, output_format, max_scripts):
    """Use research assistant to find the most relevant Buffy episodes for a paper topic based on grad bot notes"""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        console.print("[red]Error: CLAUDE_API_KEY must be set in environment variables[/red]")
        console.print("[yellow]Add CLAUDE_API_KEY=your-api-key to your .env file[/yellow]")
        return
    
    # Validate max_scripts
    if max_scripts < 1:
        console.print(f"[red]Error: --max-scripts must be at least 1, got {max_scripts}[/red]")
        return
    
    research_assistant_bot = ResearchAssistantBot(claude_api_key)
    
    try:
        # Use most recent folder if not provided
        if not notes_folder:
            notes_folder = str(get_most_recent_grad_notes_folder())
        
        console.print(f"[blue]Starting episode selection analysis...[/blue]")
        console.print(f"[blue]Notes folder: {notes_folder}[/blue]")
        console.print(f"[blue]Paper topic: {topic}[/blue]")
        console.print(f"[blue]Max scripts to copy: {max_scripts}[/blue]")
        
        # Select relevant episodes and copy script files to papers folder
        episode_list, copied_files, paper_folder = research_assistant_bot.select_relevant_episodes_and_copy_scripts(notes_folder, topic, topic_shorthand, max_scripts=max_scripts)
        
        if output_format == 'json':
            # Output raw JSON for programmatic use
            import json
            print(json.dumps(episode_list, indent=2))
        else:
            # Human-readable output
            console.print(f"\n[green]Most relevant episodes for '{topic}':[/green]")
            console.print(f"[blue]Found {len(episode_list)} relevant episodes[/blue]")
            console.print(f"[blue]Created paper folder: {paper_folder}[/blue]")
            console.print(f"[blue]Copied {len(copied_files)} script files to {paper_folder}/scripts/[/blue]\n")
            
            # Show top N episodes (the ones we copied)
            top_episodes = episode_list[:max_scripts]
            console.print(f"[green]Top {len(top_episodes)} episodes (scripts copied):[/green]")
            for i, episode in enumerate(top_episodes, 1):
                console.print(f"[yellow]{i:2d}.[/yellow] {episode}")
            
            # Show remaining episodes if any
            if len(episode_list) > max_scripts:
                console.print(f"\n[blue]Additional relevant episodes:[/blue]")
                for i, episode in enumerate(episode_list[max_scripts:], max_scripts + 1):
                    console.print(f"[dim]{i:2d}.[/dim] {episode}")
            
            # Show copied files
            if copied_files:
                console.print(f"\n[green]Script files copied to {paper_folder}/scripts/:[/green]")
                for file in copied_files:
                    console.print(f"[blue]  - {file}[/blue]")
            
            # Also output the JSON format for easy copying
            console.print(f"\n[blue]JSON format (for copying):[/blue]")
            import json
            console.print(f"[dim]{json.dumps(episode_list)}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--notes-folder', help='Path to folder containing grad bot notes (e.g., grad_notes/grad_bot_analysis_20251013_144539). If not provided, uses most recent folder.')
@click.option('--topic', required=True, help='Paper topic/thesis to evaluate notes against')
@click.option('--paper-folder', help='Path to paper folder for saving ratings (optional, saves to notes folder if not provided)')
def postdoc_bot(notes_folder, topic, paper_folder):
    """Use postdoc bot to rate the relevance of grad student notes to a paper topic
    
    Note: This command is automatically run during generate-paper-from-notes.
    Use this standalone command only if you need to run postdoc bot separately."""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        console.print("[red]Error: CLAUDE_API_KEY must be set in environment variables[/red]")
        console.print("[yellow]Add CLAUDE_API_KEY=your-api-key to your .env file[/yellow]")
        return
    
    postdoc_bot_instance = PostdocBot(claude_api_key)
    
    try:
        # Use most recent folder if not provided
        if not notes_folder:
            notes_folder = str(get_most_recent_grad_notes_folder())
        
        # Convert paper_folder to Path if provided
        target_paper_folder = Path(paper_folder) if paper_folder else None
        
        console.print(f"[blue]Starting postdoc bot rating analysis...[/blue]")
        console.print(f"[blue]Notes folder: {notes_folder}[/blue]")
        console.print(f"[blue]Paper topic: {topic}[/blue]")
        if target_paper_folder:
            console.print(f"[blue]Paper folder: {target_paper_folder}[/blue]\n")
        else:
            console.print(f"[blue]Paper folder: None (will save to notes folder)[/blue]\n")
        
        # Rate all notes and save to JSON
        postdoc_bot_instance.rate_all_notes(notes_folder, topic, target_paper_folder)
        
        save_location = target_paper_folder if target_paper_folder else notes_folder
        console.print(f"\n[green]✓ Postdoc bot rating complete![/green]")
        console.print(f"[blue]Results saved to: {save_location}/postdoc_ratings.json[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--notes-folder', help='Path to folder containing grad bot notes (e.g., grad_notes/grad_bot_analysis_20251013_144539). If not provided, uses most recent folder.')
def researcher_bot(notes_folder):
    """Use researcher bot to generate paper abstracts/proposals based on grad student notes
    
    This bot analyzes all grad student notes and generates 5-8 philosophically-driven paper abstracts
    suitable for submission to Buffy Studies conferences or journals like Slayage. Each abstract
    includes a rating based on how well-supported and novel the topic is."""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        console.print("[red]Error: CLAUDE_API_KEY must be set in environment variables[/red]")
        console.print("[yellow]Add CLAUDE_API_KEY=your-api-key to your .env file[/yellow]")
        return
    
    researcher_bot_instance = ResearcherBot(claude_api_key)
    
    try:
        # Use most recent folder if not provided
        if not notes_folder:
            notes_folder = str(get_most_recent_grad_notes_folder())
        
        console.print(f"[blue]Starting researcher bot abstract generation...[/blue]")
        console.print(f"[blue]Notes folder: {notes_folder}[/blue]\n")
        
        # Generate abstracts
        abstracts_data = researcher_bot_instance.generate_abstracts(notes_folder)
        
        # Save abstracts to JSON
        output_path = researcher_bot_instance.save_abstracts(abstracts_data, notes_folder)
        
        # Display abstracts in detail
        console.print(f"\n[green]{'='*80}[/green]")
        console.print(f"[green]Generated Paper Abstracts[/green]")
        console.print(f"[green]{'='*80}[/green]\n")
        
        abstracts = abstracts_data.get('abstracts', [])
        sorted_abstracts = sorted(abstracts, key=lambda x: x.get('rating', 0), reverse=True)
        
        for i, abstract in enumerate(sorted_abstracts, 1):
            console.print(f"[yellow]{i}. {abstract.get('title', 'Untitled')}[/yellow]")
            console.print(f"[blue]Rating: {abstract.get('rating', 0)}/100[/blue]")
            
            # Display key episodes if available
            key_episodes = abstract.get('key_episodes', [])
            if key_episodes:
                console.print(f"[dim]Key Episodes: {', '.join(key_episodes)}[/dim]")
            
            # Display philosophical frameworks if available
            frameworks = abstract.get('philosophical_frameworks', [])
            if frameworks:
                console.print(f"[dim]Frameworks: {', '.join(frameworks)}[/dim]")
            
            # Display abstract text
            abstract_text = abstract.get('abstract', 'No abstract provided')
            console.print(f"\n{abstract_text}\n")
            console.print(f"[dim]{'-'*80}[/dim]\n")
        
        console.print(f"[green]✓ All abstracts saved to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--paper-folder', required=True, help='Path to paper folder containing paper.md and scripts/ (e.g., papers/20251016_113410_nietzsche)')
def reviewer_bot(paper_folder):
    """Review an academic paper for a Buffy Studies conference
    
    This bot acts as a peer reviewer, evaluating the paper and providing either an ACCEPT or REJECT
    decision with detailed feedback. The review includes strengths, weaknesses, and if rejected,
    specific requests for changes."""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        console.print("[red]Error: CLAUDE_API_KEY must be set in environment variables[/red]")
        console.print("[yellow]Add CLAUDE_API_KEY=your-api-key to your .env file[/yellow]")
        return
    
    reviewer_bot_instance = ReviewerBot(claude_api_key)
    
    try:
        console.print(f"[blue]Starting peer review...[/blue]")
        console.print(f"[blue]Paper folder: {paper_folder}[/blue]\n")
        
        # Review the paper
        review_data = reviewer_bot_instance.review_paper(paper_folder)
        
        # Save review to JSON
        output_path = reviewer_bot_instance.save_review(review_data, paper_folder)
        
        # Display review
        decision = review_data.get('decision', 'UNKNOWN')
        decision_color = 'green' if decision == 'ACCEPT' else 'red'
        
        console.print(f"\n[{decision_color}]{'='*80}[/{decision_color}]")
        console.print(f"[{decision_color}]PEER REVIEW DECISION: {decision}[/{decision_color}]")
        console.print(f"[{decision_color}]{'='*80}[/{decision_color}]\n")
        
        # Overall assessment
        overall = review_data.get('overall_assessment', 'No assessment provided')
        console.print(f"[bold]Overall Assessment:[/bold]")
        console.print(f"{overall}\n")
        
        # Strengths
        strengths = review_data.get('strengths', [])
        if strengths:
            console.print(f"[green]Strengths:[/green]")
            for i, strength in enumerate(strengths, 1):
                console.print(f"[green]  {i}. {strength}[/green]")
            console.print()
        
        # Weaknesses
        weaknesses = review_data.get('weaknesses', [])
        if weaknesses:
            console.print(f"[yellow]Weaknesses:[/yellow]")
            for i, weakness in enumerate(weaknesses, 1):
                console.print(f"[yellow]  {i}. {weakness}[/yellow]")
            console.print()
        
        # Detailed comments
        detailed = review_data.get('detailed_comments', '')
        if detailed:
            console.print(f"[bold]Detailed Comments:[/bold]")
            console.print(f"{detailed}\n")
        
        # Script verification
        script_verification = review_data.get('script_verification', '')
        if script_verification:
            console.print(f"[bold]Script Citations:[/bold]")
            console.print(f"{script_verification}\n")
        
        # Requested changes (if rejected)
        if decision == 'REJECT':
            requested_changes = review_data.get('requested_changes', [])
            if requested_changes:
                console.print(f"[red]Requested Changes:[/red]")
                for i, change in enumerate(requested_changes, 1):
                    console.print(f"[red]  {i}. {change}[/red]")
                console.print()
        
        console.print(f"[blue]{'='*80}[/blue]")
        console.print(f"[green]✓ Full review saved to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def main():
    cli()


if __name__ == "__main__":
    main()