import os
import json
import tempfile
import toml
from datetime import datetime
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
    
    def save_paper(self, content: str, topic: str, conversation_file: str) -> Path:
        """Save generated paper to a file"""
        # Create a safe filename from the topic
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        
        # Include source conversation in filename
        conv_name = Path(conversation_file).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{conv_name}_{safe_topic}.md"
        
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
def extract_private(users, limit, output):
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
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--conversation', required=True, help='Path to conversation JSON file')
@click.option('--topic', required=True, help='Paper topic/thesis')
@click.option('--prompt-type', default='default', help='Type of system prompt to use (default, buffy)')
@click.option('--output', help='Custom output filename (optional)')
def generate_paper(conversation, topic, prompt_type, output):
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
                output_path = generator.save_paper(paper_content, topic, conversation)
            
            progress.update(task, description=f"Paper saved!")
            console.print(f"[green]✓ Paper saved to: {output_path}[/green]")
            
            # Show a preview of the paper
            console.print("\n[blue]Paper preview:[/blue]")
            preview = paper_content[:500] + "..." if len(paper_content) > 500 else paper_content
            console.print(Markdown(preview))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")



def main():
    cli()


if __name__ == "__main__":
    main()