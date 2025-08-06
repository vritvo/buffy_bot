import os
import json
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

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
        """Save messages to a JSON file"""
        if not messages:
            console.print("[yellow]No messages to save[/yellow]")
            return
        
        # Create output data structure
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
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]Saved {len(messages)} messages to {output_path}[/green]")


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


def main():
    cli()


if __name__ == "__main__":
    main()