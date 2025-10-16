# Buffy Bot - AI-Powered Conversation Analysis

Extracts Buffy the Vampire Slayer discussions from Zulip and generates academic papers using a multi-agent AI workflow.

## How It Works

1. **Extract** conversations from Zulip, split into weekly chunks
2. **Grad bots** analyze each week, extracting relevant Buffy content and quotes
3. **Generate paper** - single command that automatically:
   - Runs **postdoc bot** to rate notes for relevance (0-100)
   - Runs **research assistant** to select relevant episodes
   - Runs **professor bot** to write the paper using filtered notes and scripts

**Optional: Batch Paper Generation**
- Run **researcher bot** to generate 5-8 paper abstracts/topics
- Use `generate-paper --from-abstracts` to automatically generate papers for all abstracts

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

# 3. Generate paper (auto-runs postdoc bot + research assistant + professor bot)
uv run main.py generate-paper --topic "Your paper topic"
```

The `generate-paper` command automatically:
- Uses the most recent grad notes folder (or specify with `--notes-folder`)
- Runs **postdoc bot** to rate each week's notes for relevance
- Runs **research assistant** to select and copy relevant episode scripts
- Runs **professor bot** to write the final paper

**To use a specific grad notes folder:**
```bash
uv run main.py generate-paper \
  --notes-folder "grad_notes/grad_bot_analysis_20251013_144539" \
  --topic "Your paper topic"
```

## Key Options

### `generate-paper` (Main Command)
- `--topic`: Paper topic (required unless using `--from-abstracts`)
- `--notes-folder`: Grad notes folder (default: most recent)
- `--min-rating`: Min rating to include notes (default: 30)
- `--max-scripts`: Number of episode scripts to include (default: 5)
- `--verbatim-chat-threshold`: Rating for using verbatim transcripts (default: 70)
- `--weekly-conversations`: Path to weekly conversations folder (default: conversations/weekly)
- `--topic-shorthand`: Short identifier for folder naming
- `--from-abstracts`: Generate papers for all abstracts from `researcher-bot`
- `--min-abstract-rating`: Min abstract rating to generate papers for (default: 0, only with `--from-abstracts`)

**Batch Generation from Researcher Bot:**
```bash
# Generate papers for ALL abstracts
uv run main.py generate-paper --from-abstracts

# Generate papers only for abstracts rated 70+
uv run main.py generate-paper --from-abstracts --min-abstract-rating 70
```

When using `--from-abstracts`, the command reads `paper_abstracts.json` from the grad notes folder and generates a complete paper for each abstract (or each abstract above the rating threshold). Each paper gets its own folder in `papers/`.

**Note:** Weeks with ratings ≥ `--verbatim-chat-threshold` will use the original verbatim conversation transcript instead of summarized grad notes for higher fidelity.

### `postdoc-bot` (Optional Standalone)
Usually runs automatically during paper generation. Use standalone only if needed:
- `--topic`: Paper topic (required)
- `--notes-folder`: Grad notes folder (default: most recent)
- `--paper-folder`: Where to save ratings (default: notes folder)

### `research-assistant` (Optional Standalone)
Usually runs automatically during paper generation. Use standalone to preview episodes:
- `--topic`: Paper topic (required)
- `--notes-folder`: Grad notes folder (default: most recent)
- `--max-scripts`: Number of scripts to copy (default: 5)

### `researcher-bot` (Generate Paper Ideas)
Analyzes all grad notes and generates 5-8 philosophically-driven paper abstracts with ratings:
```bash
uv run main.py researcher-bot
# or specify folder:
uv run main.py researcher-bot --notes-folder "grad_notes/grad_bot_analysis_20251013_144539"
```
- Outputs `paper_abstracts.json` in the grad notes folder
- Each abstract includes title, 150-250 word abstract, key episodes, philosophical frameworks, and rating (0-100)
- Rating based on evidence support + novelty/interest
- Suitable for conference submissions (Slayage Conference) or journals

## Output Structure

```
papers/20251016_113410_nietzsche/
├── paper.md                    # Final paper with metadata
├── postdoc_ratings.json        # Relevance ratings (0-100)
└── scripts/                    # Top N episode scripts (configurable)
    ├── 6x17 Normal Again.txt
    └── 4x22 Restless.txt

grad_notes/grad_bot_analysis_20251013_144539/
├── 2025-07-20.md              # Weekly analysis
├── 2025-07-27.md
└── paper_abstracts.json       # Generated paper ideas (from researcher-bot)
```