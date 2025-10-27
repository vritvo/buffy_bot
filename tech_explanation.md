# Technical Documentation
## About the Slayerfest / Buffy Bot System

Slayerfest is a fully automated conference made up of fully automated academic labs — staffed entirely by AI bots — dedicated to interpreting private conversations about Buffy the Vampire Slayer and turning them into "publishable" scholarship.

The driving philosophy was:

```
If there’s a problem…
add another bot.
```

Our source material: thousands of lines of Buffy-theorizing from a real chat log.

Our goal: remove the humans who wrote those messages from the interpretive process — entirely. Except as raw data.


## System Architecture - The Bureaucracy of an AI Academic Lab

### Step 1 - The Grad Bots
Each grad bot reads one week of conversation and writes a sober academic summary, blissfully unaware of any thesis or purpose. They produce field notes, not arguments.


### Step 2 - Postdoc Bots
The postdocs are more ambitious. They receive the grad bots’ notes and a thesis statement. Their task is to rate each week’s relevance (0–100).

Scores >30/100: the week’s summary is sent to the professor.

Scores >80/100: the verbatim transcripts go to the professor (we do have a context window to worry about)

You might think this lab would never hit the context window with this process. You’d be wrong.

### Step 3 - Research Assistant bots
Reads the thesis and finds the most thematically relevant Buffy episodes. Their scripts are added to the reading pile.

### Step 4 - Professor Bot

At last, the professor. Given the postdocs’ selections of grad student summaries and ver batim chats, as well as the the RA’s episodes, the professor composes an academic paper. It is authoritative, theoretical, and unbothered by the fact that its “sources” are two people texting about Buffy.

Why send both summaries and transcripts? It’s partly a context-window optimization. And partly because academic redundancy is a sacred institution.

## The Autonomous Research Cycle

This would have been enough — but it still required a human to provide the thesis. So we removed the humans.

### Researcher Bot 
Before any lab kicks off, a researcher bot reads the chats and proposes abstract ideas and potential papers. Each abstract enters the full academic pipeline above.

### Peer Review Bot
When each paper generation process finishes, the paper is submitted to multiple Peer Reviewer Bots for review. If rejected, it returns to the lab, where the professor must revise it using the reviewer’s comments.

This continues until all papers are accepted — at which point the conference is considered complete.

## Watch a Recording of a Conference
The typical runtime of a conference is about 5 hours for 8 papers. We manage the 30k token/minute rate limit by consistently pausing for 60 seconds between submissions when necessary (the grad students need to get some rest at some point, after all). 

Below is a sped up replay of a Slayerfest conference, with all the idle waiting cut out. It runs less than 5 minutes. 
