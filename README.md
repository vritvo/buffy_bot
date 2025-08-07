# Buffy Bot - Zulip Conversation Extractor

A simple tool to extract conversations from Zulip and save them for further processing (like generating papers with LLMs).

## Features

- Extract messages from Zulip stream topics
- Extract messages from private conversations
- Simple JSON output format with message content, sender, and timestamp
- Filters out system messages, file uploads, and other non-text content
- Command-line interface with progress indicators

## Setup

1. **Install dependencies:**
   ```bash
   pip install -e .
   ```

2. **Configure environment variables:**
   Create a `.env` file in the project root with exactly this format:
   ```
   ZULIP_API_KEY=your-actual-api-key-here
   ZULIP_SITE_URL=https://your-zulip-instance.com
   ZULIP_EMAIL=your-email@example.com
   CLAUDE_API_KEY=your-claude-api-key-here
   ```

3. **Get your Zulip API key:**
   - Log into your Zulip instance in your web browser
   - Click on your profile picture in the top-right corner
   - Select "Settings" from the dropdown menu
   - In the left sidebar, click on "API keys"
   - Click "Generate API key"
   - Copy the generated API key (it will look like: `abcd1234-efgh-5678-ijkl-mnopqrstuvwx`)
   - Paste it into your `.env` file as the value for `ZULIP_API_KEY`

4. **Find your email address:**
   - In the same Zulip settings, go to "Account & privacy"
   - Copy the email address shown
   - Add it to your `.env` file as the value for `ZULIP_EMAIL`

5. **Get your Claude API key:**
   - Go to [Claude Console](https://console.anthropic.com/)
   - Sign in or create an account
   - Navigate to "API Keys" in your account settings
   - Click "Create Key" and copy the generated key
   - Add it to your `.env` file as the value for `CLAUDE_API_KEY`

## Usage

### Extract from Stream Topic
```bash
python main.py extract-stream --stream "general" --topic "daily-checkin" --limit 500
```

### Extract from Private Conversation
```bash
python main.py extract-private --users "user1@example.com,user2@example.com" --limit 1000
```

### Extract with Weekly Chunks (for Multi-LLM Processing)
```bash
python main.py extract-private --users "user1@example.com,user2@example.com" --limit 1000 --weekly-chunks
```

### Generate Academic Paper
```bash
python main.py generate-paper --conversation "conversations/private_conversation.json" --topic "Nietzschean values in Buffy's Gingerbread"
```

### Run Grad Bot Analysis on Weekly Chunks
```bash
python main.py run-grad-bots --weekly-dir "conversations/weekly" --topic "Nietzschean values in Buffy's Gingerbread" --prompt-type "grad_bot_buffy"
```

## Options

### For Message Extraction
- `--limit`: Maximum number of messages to extract (default: 1000)
- `--output`: Custom output filename (optional)
- `--weekly-chunks`: Split conversation into weekly chunks for multi-LLM processing

### For Paper Generation
- `--conversation`: Path to conversation JSON file (required)
- `--topic`: Paper topic/thesis statement (required)
- `--prompt-type`: Type of system prompt (`default` or `buffy`, default: `default`)
- `--output`: Custom output filename (optional)

### For Grad Bot Analysis
- `--weekly-dir`: Path to directory containing weekly chunk files (required)
- `--topic`: Paper topic/thesis for analysis (required)
- `--prompt-type`: Type of grad bot prompt (`grad_bot_default` or `grad_bot_buffy`, default: `grad_bot_default`)

## Output Format

Messages are saved as JSON files in the `conversations/` directory with this structure:

```json
{
  "metadata": {
    "extracted_at": "2024-01-01T12:00:00",
    "total_messages": 250,
    "first_message": 1704110400,
    "last_message": 1704196800
  },
  "messages": [
    {
      "content": "Hello world!",
      "sender": "John Doe",
      "timestamp": 1704110400
    }
  ]
}
```

## What Gets Extracted

- ✅ Text messages with full content (including markdown)
- ✅ User names and timestamps
- ❌ Message IDs, stream/topic information, and other metadata
- ❌ System messages
- ❌ File uploads
- ❌ Images
- ❌ Emoji reactions

## Examples

### Working with your conversation
Based on your usage, to extract the conversation from your URL like `https://recurse.zulipchat.com/#narrow/dm/896241-frederic-kettelhoit-(he)-(F1'25)`, use:

```bash
python main.py extract-private --users "kettelhoit@gmail.com" --limit 1000
```

This will save the conversation to `conversations/private_conversation.json` ready for further processing.

### Generate a paper from your conversation
Once you have extracted the conversation, generate an academic paper:

```bash
python main.py generate-paper --conversation "conversations/private_conversation.json" --topic "Nietzschean values in Buffy's Gingerbread" --prompt-type "buffy"
```

## Paper Generation Features

- **Multiple prompt types**: Choose between `default` (general academic writing) or `buffy` (specialized for Buffy analysis with philosophical theory)
- **Configurable prompts**: Edit `prompts.toml` to customize the system prompts
- **Automatic formatting**: Papers are saved as Markdown files with metadata headers
- **Smart filenames**: Output files include timestamp, source conversation, and topic
- **Preview**: Shows a preview of the generated paper in the terminal

### System Prompts

The tool includes two preconfigured system prompts:

1. **`default`**: General academic writing focused on media studies and cultural analysis
2. **`buffy`**: Specialized for Buffy the Vampire Slayer analysis, including expertise in:
   - Nietzschean and Freudian theory
   - Feminist and queer studies
   - Buffy scholarship and character analysis
   - Symbolic and metaphorical interpretation

You can customize these prompts by editing the `prompts.toml` file.

## Multi-LLM Processing with Weekly Chunks

The `--weekly-chunks` flag enables a workflow for processing large conversations with multiple LLM "grad students":

### How Weekly Chunking Works

1. **Time-based Division**: Messages are split into weekly chunks based on Unix timestamps
2. **Sunday 3am ET Cutoff**: Each week starts on Sunday at 3am Eastern Time
3. **Separate Files**: Each week is saved as a separate `.txt` file in `conversations/weekly/`
4. **Ready for Processing**: Each chunk is sized appropriately for individual LLM analysis

### Example Output Structure
```
conversations/
├── private_conversation.json          # Full conversation
├── private_conversation.txt           # All content in one file
└── weekly/
    ├── private_conversation_week_2024-01-07.txt
    ├── private_conversation_week_2024-01-14.txt
    ├── private_conversation_week_2024-01-21.txt
    └── ...
```

### Workflow for Multiple LLMs
1. Extract with weekly chunks: `--weekly-chunks`
2. Send each weekly file to a different LLM "grad student" for analysis
3. Aggregate the insights from each week
4. Generate final paper from combined analysis

This approach allows for:
- **Parallel processing** of different time periods
- **Focused analysis** on specific weeks
- **Better token management** for large conversations
- **Distributed workload** across multiple LLM instances

## Grad Bot Research Assistant Workflow

The grad bot system provides AI research assistants that analyze weekly conversation chunks and extract relevant quotes and notes for paper writing.

### How Grad Bots Work

1. **Sequential Processing**: Each weekly chunk is analyzed by a grad bot research assistant
2. **Quote Extraction**: Grad bots extract EXACT quotes relevant to your paper topic
3. **Analytical Notes**: Brief explanatory notes about why quotes are relevant
4. **Focused Analysis**: Each grad bot specializes in finding substantive content, not casual mentions
5. **Quality over Quantity**: Grad bots only extract truly relevant material

### Grad Bot Types

1. **`grad_bot_default`**: General media studies and cultural analysis
2. **`grad_bot_buffy`**: Specialized for Buffy the Vampire Slayer analysis with theoretical knowledge

### Complete Multi-LLM Workflow

```bash
# Step 1: Extract conversation with weekly chunks
python main.py extract-private --users "user@example.com" --limit 1000 --weekly-chunks

# Step 2: Run grad bots on all weekly chunks
python main.py run-grad-bots --weekly-dir "conversations/weekly" --topic "Your Paper Topic" --prompt-type "grad_bot_buffy"

# Step 3: Review grad bot notes and generate final paper
# (Manual review of grad_notes/ directory, then use insights for paper generation)
```

### Grad Bot Output Structure

```
grad_notes/
├── 20250106_120000_private_conversation_week_2024-07-20_Your_Topic_notes.md
├── 20250106_120130_private_conversation_week_2024-07-27_Your_Topic_notes.md
├── 20250106_120245_private_conversation_week_2024-08-03_Your_Topic_notes.md
└── ...
```

Each grad bot note file contains:
- **Metadata**: Paper topic, source week, analysis timestamp
- **Relevant Quotes**: Exact conversation pieces related to the topic
- **Analytical Notes**: Brief explanations of relevance and connections
- **Week Summary**: Overall assessment of the week's relevance

### Benefits of Grad Bot Workflow

- **Systematic Analysis**: Every week gets thorough examination
- **Exact Citations**: Preserves original conversation text for accurate quoting
- **Focused Research**: Only extracts content relevant to your specific topic
- **Scalable Processing**: Can handle large conversations across many weeks
- **Research Foundation**: Creates organized material for final paper writing