# Technical Documentation

## About the Buffy Bot System

The Buffy Bot is an AI-powered conversation analysis and academic paper generation system. It extracts discussions about *Buffy the Vampire Slayer* from Zulip chat logs and generates academic papers using a multi-agent AI workflow.

## System Architecture

### 1. Conversation Extraction

The system extracts private conversations from Zulip and optionally breaks them into weekly chunks for detailed analysis.

```bash
uv run main.py extract-private --limit 10000 --weekly-chunks
```

### 2. Graduate Bot Analysis

Graduate-level AI agents analyze the weekly conversation chunks, identifying key themes, philosophical concepts, and noteworthy observations.

```bash
uv run main.py run-grad-bots --weekly-dir "conversations/weekly"
```

### 3. Research Bot

The researcher bot synthesizes the graduate notes to propose 5-8 compelling paper topics, each with:
- A detailed abstract
- Evidence assessment (0-100 rating)
- Key episode references
- Novelty evaluation

```bash
uv run main.py researcher-bot
```

### 4. Conference Simulation

The `run-conference` command fully automates the academic lifecycle:

```bash
uv run main.py run-conference
```

This command:
- Generates papers for all highly-rated abstracts
- Reviews each paper with multiple AI peer reviewers
- Rewrites papers based on feedback
- Continues revision cycles until all reviews are ACCEPT
- Features realistic academic filename chaos (paper_copy1_FINAL2.md, etc.)
- Includes acceptance bias that increases with longer filenames (simulating academic reality)

## Paper Generation Pipeline

### Postdoc Bot
Rates each weekly note for relevance to the paper topic (0-100 scale). Only weeks meeting the threshold are included.

### Research Assistant
Selects the most relevant episode scripts based on the topic and conversation analysis.

### Professor Bot
Writes the academic paper using:
- Filtered weekly notes (or verbatim transcripts for highly relevant weeks)
- Selected episode scripts
- Theoretical frameworks appropriate to the topic

## Output Structure

Each paper folder contains:
- `paper.md` / `paper_final1.md` - The paper in Markdown format
- `paper.pdf` / `paper_final1.pdf` - PDF version
- `postdoc_ratings.json` - Relevance ratings for each week
- `reviews/` - Peer review JSON files
- `scripts/` - Episode script files used

Revised papers get version suffixes (`_v2`, `_v2_v2`) and accumulate increasingly chaotic filenames.

## Configuration

Key parameters are configurable in `prompts.toml`:
- `max_weeks_for_paper` - Maximum weeks to include (default: 8)
- Agent prompts and system messages
- Review criteria and standards

## Technologies Used

- **Claude Sonnet 4**: All AI agents use Claude for text generation
- **Python**: Core system implementation
- **uv**: Dependency management
- **Rich**: Terminal UI
- **Markdown & PDF**: Paper output formats
- **Zulip API**: Conversation extraction

## Static Site Generation

This website was generated using `static_site_generator.py`, which:
- Scans all paper folders in the `papers/` directory
- Identifies the final accepted version of each paper series
- Generates responsive HTML pages with embedded CSS
- Creates individual pages for papers and reviews
- Copies PDFs for download

```bash
uv run static_site_generator.py --papers-dir papers --output-dir conference_site
```

## Source Code

The complete source code is available in the project repository. Key files:
- `main.py` - Core CLI and agent orchestration
- `prompts.toml` - Agent prompts and configuration
- `static_site_generator.py` - Website generation
- `grad_bot_logger.py` - Logging utilities

