import os
import json
import tempfile
import toml
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from dotenv import load_dotenv
from anthropic import Anthropic

from zulip import Client

# Load environment variables
load_dotenv()

console = Console()


@dataclass
class Message:
    """Simple message data structure"""
    content: str
    sender: str
    timestamp: str


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
                    timestamp=msg['timestamp']
                )
                messages.append(message)
        
        return messages
    
    def extract_private_messages(self, user_emails: List[str], limit: int = 1000) -> List[Message]:
        """Extract messages from a private conversation using email addresses"""
        messages = []
        
        # For private messages, we can use the email directly in the narrow
        # The Zulip API accepts email addresses for pm-with operator
        request = {
            'anchor': 'newest',
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
                    timestamp=msg['timestamp']
                )
                messages.append(message)
        
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
    
    def load_system_prompt(self, prompt_type: str = "default") -> str:
        """Load system prompt from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
        if prompt_type not in config:
            available = list(config.keys())
            raise ValueError(f"Prompt type '{prompt_type}' not found. Available: {available}")
        
        return config[prompt_type]["system_prompt"]
    
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
    
    def generate_paper(self, conversation_file: str, topic: str, prompt_type: str = "default") -> str:
        """Generate a paper using Claude based on the conversation and topic"""
        
        console.print(f"[blue]Loading conversation from: {conversation_file}[/blue]")
        transcript = self.load_conversation(conversation_file)
        
        console.print(f"[blue]Loading system prompt: {prompt_type}[/blue]")
        system_prompt = self.load_system_prompt(prompt_type)
        
        user_prompt = f"""Please write an academic paper on the topic: "{topic}"

Based on the following chat transcript:

{transcript}

The paper should be well-structured, insightful, and grounded in the evidence and discussion from the transcript."""
        
        console.print("[yellow]Sending request to Claude...[/yellow]")
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
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
    
    def save_paper(self, content: str, topic: str, conversation_file: str, topic_shorthand: str = None) -> Path:
        """Save generated paper to a file"""
        # Use provided shorthand or fallback to a simple default
        if topic_shorthand:
            # Clean the shorthand to be filesystem-safe
            safe_shorthand = "".join(c for c in topic_shorthand if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_shorthand = safe_shorthand.replace(' ', '_')
        else:
            # Fallback: use first few words if no shorthand provided
            topic_words = topic.split()[:3]
            safe_shorthand = "_".join(word for word in topic_words if word.isalnum())[:30]
        
        # Include source conversation in filename
        conv_name = Path(conversation_file).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{conv_name}_{safe_shorthand}.md"
        
        output_path = self.output_dir / filename
        
        # Add metadata header to the paper
        metadata = f"""---
title: "{topic}"
source_conversation: "{conversation_file}"
generated_at: "{datetime.now().isoformat()}"
model: "claude-3-5-sonnet-20241022"
---

"""
        
        full_content = metadata + content
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return output_path


class GradBot:
    """Graduate student research assistant for analyzing weekly conversation chunks"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.output_dir = Path("grad_notes")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_grad_bot_prompts(self, prompt_type: str = "grad_bot_default") -> tuple[str, str]:
        """Load grad bot system and user prompts from configuration file"""
        config_path = Path("prompts.toml")
        if not config_path.exists():
            raise FileNotFoundError("prompts.toml configuration file not found")
        
        config = toml.load(config_path)
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
            user_prompt = """Paper topic: "{{PAPER_TOPIC}}"

Please analyze the following weekly conversation transcript and extract relevant content for this paper topic:

{{CONVERSATION_TRANSCRIPT}}

Remember to:
- Extract EXACT quotes that are relevant to the paper topic
- Provide brief analytical notes for each quote
- Note if this week has little/no relevant content (that's perfectly fine)
- Focus on substance and relevance over quantity"""
        
        return system_prompt, user_prompt
    
    def analyze_weekly_chunk(self, weekly_file: str, paper_topic: str, prompt_type: str = "grad_bot_default") -> str:
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
        system_prompt, user_prompt_template = self.load_grad_bot_prompts(prompt_type)
        
        # Substitute variables in user prompt
        user_prompt = user_prompt_template.replace("{{PAPER_TOPIC}}", paper_topic)
        user_prompt = user_prompt.replace("{{CONVERSATION_TRANSCRIPT}}", weekly_content)
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
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
    
    def save_grad_notes(self, analysis: str, weekly_file: str, paper_topic: str, topic_shorthand: str = None) -> Path:
        """Save grad bot analysis to a file"""
        # Extract week identifier from filename
        week_file = Path(weekly_file)
        week_name = week_file.stem  # e.g., "private_conversation_week_2024-01-07"
        
        # Use provided shorthand or fallback to a simple default
        if topic_shorthand:
            # Clean the shorthand to be filesystem-safe
            safe_shorthand = "".join(c for c in topic_shorthand if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_shorthand = safe_shorthand.replace(' ', '_')
        else:
            # Fallback: use first few words if no shorthand provided
            topic_words = paper_topic.split()[:3]
            safe_shorthand = "_".join(word for word in topic_words if word.isalnum())[:30]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{week_name}_{safe_shorthand}_notes.md"
        
        output_path = self.output_dir / filename
        
        # Add metadata header
        metadata = f"""---
paper_topic: "{paper_topic}"
source_week: "{weekly_file}"
analyzed_at: "{datetime.now().isoformat()}"
model: "claude-3-5-sonnet-20241022"
grad_bot_type: "research_assistant"
---

# Grad Bot Analysis: {week_name}

**Paper Topic:** {paper_topic}

**Source:** {weekly_file}

---

{analysis}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(metadata)
        
        return output_path

    def process_all_weekly_chunks(self, weekly_dir: str, paper_topic: str, prompt_type: str = "grad_bot_default", topic_shorthand: str = None):
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
                    analysis = self.analyze_weekly_chunk(str(weekly_file), paper_topic, prompt_type)
                    
                    # Save the analysis
                    output_path = self.save_grad_notes(analysis, str(weekly_file), paper_topic, topic_shorthand)
                    
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
        console.print(f"[blue]Notes saved to: {self.output_dir}[/blue]")
        
        return results


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
@click.option('--users', required=True, help='Comma-separated list of user emails')
@click.option('--limit', default=1000, help='Maximum number of messages to extract')
@click.option('--output', help='Output filename (default: private_conversation.json)')
@click.option('--weekly-chunks', is_flag=True, help='Also split conversation into weekly chunks for multi-LLM processing')
def extract_private(users, limit, output, weekly_chunks):
    """Extract messages from a private conversation using email addresses"""
    api_key = os.getenv('ZULIP_API_KEY')
    site_url = os.getenv('ZULIP_SITE_URL')
    
    if not api_key or not site_url:
        console.print("[red]Error: ZULIP_API_KEY and ZULIP_SITE_URL must be set in environment variables[/red]")
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
@click.option('--prompt-type', default='default', help='Type of system prompt to use (default, buffy)')
@click.option('--output', help='Custom output filename (optional)')
@click.option('--topic-shorthand', help='Short identifier for the topic (used in filenames if --output not provided)')
def generate_paper(conversation, topic, prompt_type, output, topic_shorthand):
    """Generate an academic paper from a conversation transcript using Claude"""
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
            paper_content = generator.generate_paper(conversation, topic, prompt_type)
            progress.update(task, description="Saving paper...")
            
            # Save the paper
            if output:
                # If custom output provided, use it
                output_path = Path("papers") / output
                if not output.endswith('.md'):
                    output_path = output_path.with_suffix('.md')
                output_path.parent.mkdir(exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(paper_content)
            else:
                # Use automatic filename
                output_path = generator.save_paper(paper_content, topic, conversation, topic_shorthand)
            
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
@click.option('--topic', required=True, help='Paper topic/thesis for analysis')
@click.option('--prompt-type', default='grad_bot_default', help='Type of grad bot prompt (grad_bot_default, grad_bot_buffy)')
@click.option('--topic-shorthand', required=True, help='Short identifier for the topic (used in filenames)')
def run_grad_bots(weekly_dir, topic, prompt_type, topic_shorthand):
    """Run grad bot analysis on all weekly conversation chunks"""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        console.print("[red]Error: CLAUDE_API_KEY must be set in environment variables[/red]")
        console.print("[yellow]Add CLAUDE_API_KEY=your-api-key to your .env file[/yellow]")
        return
    
    grad_bot = GradBot(claude_api_key)
    
    try:
        console.print(f"[blue]Starting grad bot analysis...[/blue]")
        console.print(f"[blue]Topic: {topic}[/blue]")
        console.print(f"[blue]Weekly directory: {weekly_dir}[/blue]")
        console.print(f"[blue]Prompt type: {prompt_type}[/blue]")
        
        results = grad_bot.process_all_weekly_chunks(weekly_dir, topic, prompt_type, topic_shorthand)
        
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
                console.print(f"[yellow]2. Use the extracted quotes and notes to generate your final paper[/yellow]")
            
            failed_count = sum(1 for r in results if not r['success'])
            if failed_count > 0:
                console.print(f"\n[red]Failed to process {failed_count} weeks[/red]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def main():
    cli()


if __name__ == "__main__":
    main()