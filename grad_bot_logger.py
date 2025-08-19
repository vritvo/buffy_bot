import os
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class GradBotRun:
    """Data structure for a grad bot run record"""
    date: str
    prompt_hash: str
    model: str
    temperature: float
    max_tokens: int
    result_folder: str
    rating: str = ""
    notes: str = ""
    changes_from_previous: str = ""


class GradBotLogger:
    """Handles logging of grad bot prompts and run metadata"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.logs_dir = self.base_dir / "logs"
        self.prompts_dir = self.logs_dir / "prompts"
        self.csv_path = self.logs_dir / "grad_bot_runs.csv"
        
        # Ensure directories exist
        self.logs_dir.mkdir(exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Initialize CSV file if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            headers = [
                "date",
                "prompt_hash", 
                "model",
                "temperature",
                "max_tokens",
                "result_folder",
                "rating",
                "notes", 
                "changes_from_previous"
            ]
            
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def _hash_prompt(self, system_prompt: str, user_prompt_template: str) -> str:
        """Generate a hash for the combined prompt content"""
        # Combine both prompts and hash them
        combined_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER_TEMPLATE:\n{user_prompt_template}"
        
        # Create SHA-256 hash
        hash_object = hashlib.sha256(combined_prompt.encode('utf-8'))
        return hash_object.hexdigest()[:16]  # Use first 16 characters for readability
    
    def log_prompt(self, system_prompt: str, user_prompt_template: str) -> str:
        """Log the prompt to a file and return the hash"""
        prompt_hash = self._hash_prompt(system_prompt, user_prompt_template)
        prompt_file = self.prompts_dir / f"{prompt_hash}.txt"
        
        # Only write if file doesn't exist (avoid duplicates)
        if not prompt_file.exists():
            combined_prompt = f"""=== GRAD BOT PROMPT LOG ===
Timestamp: {datetime.now().isoformat()}
Hash: {prompt_hash}

=== SYSTEM PROMPT ===
{system_prompt}

=== USER PROMPT TEMPLATE ===
{user_prompt_template}

=== END PROMPT LOG ==="""
            
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(combined_prompt)
        
        return prompt_hash
    
    def log_run(self, run_data: GradBotRun):
        """Log a grad bot run to the CSV file"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_data.date,
                run_data.prompt_hash,
                run_data.model,
                run_data.temperature,
                run_data.max_tokens,
                run_data.result_folder,
                run_data.rating,
                run_data.notes,
                run_data.changes_from_previous
            ])
    
    def log_grad_bot_execution(
        self, 
        system_prompt: str, 
        user_prompt_template: str,
        model: str,
        temperature: float,
        max_tokens: int,
        result_folder_name: str
    ) -> str:
        """Complete logging for a grad bot execution"""
        # Log the prompt and get hash
        prompt_hash = self.log_prompt(system_prompt, user_prompt_template)
        
        # Create run record
        run_data = GradBotRun(
            date=datetime.now().isoformat(),
            prompt_hash=prompt_hash,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            result_folder=result_folder_name
        )
        
        # Log the run
        self.log_run(run_data)
        
        return prompt_hash
    
    def get_prompt_by_hash(self, prompt_hash: str) -> str:
        """Retrieve a prompt by its hash"""
        prompt_file = self.prompts_dir / f"{prompt_hash}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def list_recent_runs(self, limit: int = 10) -> list:
        """Get the most recent grad bot runs"""
        if not self.csv_path.exists():
            return []
        
        runs = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            runs = list(reader)
        
        # Return most recent runs (last `limit` entries)
        return runs[-limit:] if len(runs) > limit else runs
