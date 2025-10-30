# Slayerfest '03 The Buffy Bot Conference

**Website** https://slayerfest.org
**Summary** Fully automated AI academic conference on *Buffy the Vampire Slayer*
**Collaborators** Built with [Frederic Kettelhoit](https://github.com/fkettelhoit/)
## What This is

Slayerfest / Buffy Bot is a performance art codebase disguised as an academic workflow. It runs a full AI conference — complete with grad students, post docs, peer reviewers, and professors — all bots, no humans (except the original chat log they're sourcing from).

The multi-agent system is fully vibe-coded. It ingests thousands of lines of *Buffy* discussion from a real conversation, runs it through a bureacratic multi-agent process, and outputs a "conference proceeding" website with papers, reviews, and revision histories. 

| Bot           | Description                               |
| -----------   | ------------------------------------------|
|Researcher Bot |Generates potential abstracts and paper ideas|
|Grad Bot              | Summarize weekly conversations (no thesis)|
|Post Doc Bot          |Rate weekly summaries (0–100) against a thesis|
|Research Assistant Bot|Selects the most relevant Buffy episode scripts|
|Professor Bot   |Writes the academic paper|
|Peer Reviewer Bots   |Review and force revisions until all papers are accepted|


You can also learn more about what each bot does on the technical documentation page of the website [here](https://slayerfest.org/technical_documentation).

Each conference run automatically produces papers, reviews, and versioned filenames with increasing chaos (e.g. `paper_FINAL_v2_copy3.md`).

## How to Run A Conference
### 1. Set up

You first need to set up some details in your .env file: 

```
ZULIP_API_KEY=your-zulip-api-key
ZULIP_SITE_URL=https://your-zulip-instance.com
ZULIP_EMAIL=your-email@example.com #email you use for zulip
ZULIP_RECIPIENT=other-person@example.com #email of the other person in the 1:1 conversation
CLAUDE_API_KEY=your-claude-api-key
```

This project uses `uv` for dependency management. You can synv with:

```
uv sync
```

### 2. Run a conference

```
# 1. Extract conversations
uv run main.py extract-private --limit 10000 --weekly-chunks

# 2. Analyze with grad bots
uv run main.py run-grad-bots --weekly-dir "conversations/weekly"

# 3. Generate paper abstracts (5-8 philosophically-driven topics)
uv run main.py researcher-bot

# 4. Run the full conference (generates papers, reviews, revisions until acceptance)
uv run main.py run-conference
```

There are some key commands for `run-conference`, which may significantly impact the output:

--notes-folder: Grad notes folder (default: most recent)
--min-abstract-rating: Min abstract rating to include (default: 0)
--num-reviews: Reviewers per round (default: 3)
--max-revision-rounds: Max revision attempts before giving up (default: 10)
--min-rating: Min postdoc rating to include notes (default: 30)
--max-scripts: Episode scripts to include (default: 5)
--verbatim-chat-threshold: Rating for using verbatim transcripts (default: 70)

A full conference can take many hours, because there are many 60 second breaks baked in to manage the rate limits. Below is a 10 minute version of the bots-at-work, with all the rests cut out: 

<img src="./videos/conference_run.gif" alt="Conference Run" width="400">

## Website

Once the papers are ready: 

```
uv run python static_site_generator.py 
```

This takes source files and builds the website from it. 

Key options:

--papers-dir: Path to papers directory (default: papers)
--output-dir: Output directory for static site (default: conference_site)
--landing-page: Markdown file for landing page content (required, default: landing_page.md)
--tech-docs: Markdown file for technical documentation (required, default: tech_explanation.md)
--svg-file: svg file for website overlay
--custom-css: Optional custom CSS file to use instead of default styling

## Other Options

There are some other options here that let you run parts of the pipeline, without a full conference. For instance, you can run a "lab" to generate a paper for a human-provided abstract with 


```
# 1. Extract and run gard student steps above
# 2. Generate paper (auto-runs postdoc bot + research assistant + professor bot)
uv run main.py generate-paper --topic "Your paper topic"
```

The `generate-paper` command automatically:
- Runs **postdoc bot** to rate notes for relevance (0-100)
- Runs **research assistant** to select relevant episodes
- Runs **professor bot** to write the paper using filtered notes and scripts
