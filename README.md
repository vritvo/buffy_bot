# Buffy Bot - AI-Powered Conversation Analysis

Extracts Buffy the Vampire Slayer discussions from Zulip and generates academic papers using a multi-agent AI workflow.

## How It Works

1. **Extract** conversations from Zulip, split into weekly chunks
2. **Grad bots** analyze each week, extracting relevant Buffy content and quotes
3. **Postdoc bot** rates each week's notes for relevance to your paper topic (0-100)
4. **Research assistant** identifies the most relevant episodes and copies their scripts
5. **Professor bot** writes the final paper using filtered notes (min rating threshold) and episode scripts

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

### Multi-Agent Workflow

```bash
# 1. Extract conversations
uv run main.py extract-private --limit 10000 --weekly-chunks

# 2. Analyze with grad bots
uv run main.py run-grad-bots --weekly-dir "conversations/weekly"

# 3. Rate notes for relevance
uv run main.py postdoc-bot --topic "Your paper topic"

# 4. Generate paper (auto-runs research assistant + professor bot)
uv run main.py generate-paper-from-notes --topic "Your paper topic"
```

The `generate-paper-from-notes` command automatically uses the most recent grad notes folder and runs both the research assistant (to select relevant episodes) and professor bot (to write the paper).

**To use a specific grad notes folder:**
```bash
# Specify notes folder for postdoc and paper generation
uv run main.py postdoc-bot \
  --notes-folder "grad_notes/grad_bot_analysis_20251013_144539" \
  --topic "Your paper topic"

uv run main.py generate-paper-from-notes \
  --notes-folder "grad_notes/grad_bot_analysis_20251013_144539" \
  --topic "Your paper topic"
```

## Key Options

### `postdoc-bot`
- `--topic`: Paper topic (required)
- `--notes-folder`: Grad notes folder (default: most recent)

Creates `postdoc_ratings.json` with relevance scores (0-100) for each week.

### `generate-paper-from-notes`
- `--topic`: Paper topic (required)
- `--notes-folder`: Grad notes folder (default: most recent)
- `--min-rating`: Min rating to include notes (default: 30)
- `--max-scripts`: Number of episode scripts to include (default: 5)
- `--topic-shorthand`: Short identifier for folder naming

### `research-assistant`
Standalone episode selection (auto-runs in `generate-paper-from-notes`):
- `--topic`: Paper topic (required)
- `--notes-folder`: Grad notes folder (default: most recent)
- `--max-scripts`: Number of scripts to copy (default: 5)

## Output Structure

```
papers/20251016_113410_nietzsche/
├── paper.md                    # Final paper with metadata
└── scripts/                    # Top N episode scripts (configurable)
    ├── 6x17 Normal Again.txt
    └── 4x22 Restless.txt

grad_notes/grad_bot_analysis_20251013_144539/
├── 2025-07-20.md              # Weekly analysis
├── 2025-07-27.md
└── postdoc_ratings.json       # Relevance ratings (0-100)
```