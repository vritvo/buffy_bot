# Buffy Bot - AI-Powered Conversation Analysis

A tool for extracting Buffy the Vampire Slayer discussions from Zulip conversations and generating academic papers using AI analysis.

## How It Works

This workflow uses multiple AI agents to analyze conversations about Buffy episodes:

1. **Extract conversations** from Zulip and split into weekly chunks
2. **Grad bots** analyze each week independently, extracting relevant Buffy content and quotes
3. **Research assistant** identifies the most relevant episodes for your thesis and copies their scripts
4. **Professor bot** writes the final academic paper using the curated analysis and episode scripts

## Setup

This project uses `uv` for dependency management. Set up your environment variables in a `.env` file:

```
ZULIP_API_KEY=your-zulip-api-key
ZULIP_SITE_URL=https://your-zulip-instance.com
ZULIP_EMAIL=your-email@example.com #email you use for zulip
ZULIP_RECIPIENT=other-person@example.com #email of the other person in the 1:1 conversation
CLAUDE_API_KEY=your-claude-api-key
```

## Usage

### Complete Workflow

```bash
# Step 1: Extract conversation with weekly chunks
uv run python main.py extract-private --limit 1000 --weekly-chunks

# Step 2: Run grad bots on all weekly chunks
uv run python main.py run-grad-bots --weekly-dir "conversations/weekly" \
  --topic "This paper examines how Buffy the Vampire Slayer's episode 'Gingerbread' (3.11), paired with 'Amends' (3.10), functions as a sophisticated philosophical treatise that weaves together Nietzschean ethics, Freudian psychoanalysis, and feminist-queer theory" \
  --prompt-type "grad_bot_buffy" --topic-shorthand "nietzsche"

# Step 3: Research assistant finds relevant episodes and creates paper folder
uv run python main.py research-assistant --notes-folder "grad_notes/nietzsche_20250827_112230" \
  --topic "This paper examines how Buffy the Vampire Slayer's episode 'Gingerbread' (3.11), paired with 'Amends' (3.10), functions as a sophisticated philosophical treatise that weaves together Nietzschean ethics, Freudian psychoanalysis, and feminist-queer theory" \
  --topic-shorthand "nietzsche"

# Step 4: Generate final paper
uv run python main.py generate-paper-from-notes --notes-folder "grad_notes/nietzsche_20250827_112230" \
  --topic "This paper examines how Buffy the Vampire Slayer's episode 'Gingerbread' (3.11), paired with 'Amends' (3.10), functions as a sophisticated philosophical treatise that weaves together Nietzschean ethics, Freudian psychoanalysis, and feminist-queer theory" \
  --paper-folder "papers/20250912_120000_nietzsche"
```

### Folder Structure

```
papers/
└── 20250912_120000_nietzsche/    # Paper folder (created by research assistant)
    ├── paper.md                  # Final paper
    └── scripts/                  # Relevant episode scripts
        ├── 2x22 Becoming Part 2.txt
        └── 3x11 Gingerbread.txt

grad_notes/
└── nietzsche_20250827_112230/    # Grad bot analysis
    ├── 2025-07-20.md
    └── 2025-07-27.md

conversations/
├── private_conversation.txt      # Full conversation
└── weekly/                       # Weekly chunks for grad bots
    ├── private_conversation_week_2025-07-20.txt
    └── private_conversation_week_2025-07-27.txt
```

## Options

### Extract Private Conversation
- `--limit`: Maximum number of messages (default: 1000)
- `--weekly-chunks`: Split into weekly chunks for multi-AI processing

### Run Grad Bots
- `--weekly-dir`: Directory containing weekly chunk files (required)
- `--topic`: Paper topic/thesis (required)
- `--prompt-type`: `grad_bot_buffy` for Buffy analysis (default: `grad_bot_buffy`)
- `--topic-shorthand`: Short identifier for folder naming (required)

### Research Assistant
- `--notes-folder`: Path to grad bot notes folder (required)
- `--topic`: Paper topic/thesis (required)
- `--topic-shorthand`: Short identifier for folder naming (optional)
- `--output-format`: `list` for human-readable or `json` (default: `list`)

### Generate Paper from Notes
- `--notes-folder`: Path to grad bot notes folder (required)
- `--topic`: Paper topic/thesis (required)
- `--prompt-type`: `default` or `buffy` (default: `default`)
- `--paper-folder`: Path to existing paper folder (optional)
- `--output`: Custom output filename (optional)