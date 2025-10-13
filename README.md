# Buffy Bot - AI-Powered Conversation Analysis

A tool for extracting Buffy the Vampire Slayer discussions from Zulip conversations and generating academic papers using AI analysis.

## How It Works

This workflow uses multiple AI agents to analyze conversations about Buffy episodes:

1. **Extract conversations** from Zulip and split into weekly chunks
2. **Grad bots** analyze each week independently, extracting relevant Buffy content and quotes
3. **Generate paper** - automatically generates paper by running
3a **Research assistant** identifies the most relevant episodes for your thesis and copies their scripts
3b **Professor bot** writes the final academic paper using the curated analysis and episode scripts

**Alternative, Direct Approach:**
1. **Extract conversations** from Zulip
2. **Generate paper** directly from raw conversation transcript

Note that this alternative "direct" approach quickly runs up against the Claude rate limit, which is why the multi-agent approach is preferred.

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
uv run python main.py run-grad-bots --weekly-dir "conversations/weekly"

# Step 3: Generate final paper (automatically runs research assistant + professor bot)
uv run python main.py generate-paper-from-notes \
  --notes-folder "grad_notes/grad_bot_analysis_20250827_112230" \
  --topic "This paper examines how Buffy the Vampire Slayer's episode 'Gingerbread' (3.11), paired with 
  'Amends' (3.10), functions as a sophisticated philosophical treatise that weaves together Nietzschean 
  ethics, Freudian psychoanalysis, and feminist-queer theory" \
  --topic-shorthand "nietzsche"
```

### Direct Approach

```bash
# Step 1: Extract conversation (no weekly chunks needed)
uv run python main.py extract-private --limit 1000

# Step 2: Generate paper directly from transcript
uv run python main.py generate-paper \
  --conversation "conversations/private_conversation.txt" \
  --topic "This paper examines how Buffy the Vampire Slayer's episode 'Gingerbread' (3.11), paired with 
  'Amends' (3.10), functions as a sophisticated philosophical treatise that weaves together Nietzschean 
  ethics, Freudian psychoanalysis, and feminist-queer theory" \
  --topic-shorthand "nietzsche"
```

### Folder Structure

```
papers/
└── 20250912_120000_nietzsche/    # Paper folder (auto-created)
    ├── paper.md                  # Final paper
    └── scripts/                  # Top 5 relevant episode scripts
        ├── 2x22 Becoming Part 2.txt
        └── 3x11 Gingerbread.txt

grad_notes/
└── grad_bot_analysis_20250827_112230/    # Grad bot analysis 
    ├── 2025-07-20.md
    └── 2025-07-27.md

conversations/
├── private_conversation.txt      # Full conversation
└── weekly/                       # Weekly chunks (multi-bot workflow only)
    ├── private_conversation_week_2025-07-20.txt
    └── private_conversation_week_2025-07-27.txt
```

## Key Options

### Extract Private Conversation
- `--limit`: Maximum number of messages (default: 1000)
- `--weekly-chunks`: Split into weekly chunks for multi-bot workflow

### Run Grad Bots
- `--weekly-dir`: Directory containing weekly chunk files

### Generate Paper from Notes (Multi-Bot)
- `--notes-folder`: Path to grad bot notes folder
- `--topic`: Paper topic/thesis
- `--topic-shorthand`: Short identifier for folder naming

### Generate Paper (Legacy Direct)
- `--conversation`: Path to conversation file
- `--topic`: Paper topic/thesis
- `--topic-shorthand`: Short identifier for folder naming