# About the Slayerfest / Buffy Bot System

Slayerfest is a fully automated conference made up of fully automated academic labs — staffed entirely by AI bots — dedicated to interpreting private conversations about *Buffy the Vampire Slayer* and turning them into "publishable" scholarship.

The driving philosophy was:

```
If there’s a problem…
add another bot.
```

Our source material: thousands of lines of *Buffy*-theorizing from a real chat log.

Our goal: remove the humans who wrote those messages from the interpretive process — entirely. Except as raw data.

## Watch a Conference Run
Below is a full conference conducted entirely by a network of academic bots.

```Conference Video```

Each line of scrolling text represents a different part of an artificial academic bureaucracy: graduate students summarizing conversations, postdocs rating them for relevance, research assistants gathering sources, a professor bot composing papers, and reviewers deciding which ones to “accept.”

This version is condensed to five minutes. The real version takes roughly five hours to produce eight papers, with built in 60-second pauses for rate limits — and to let the grad students get some rest. 

## The Bureaucracy of an AI Academic Lab

### Step 1 - The Grad Bots
Each grad bot reads one week of conversation and writes a sober academic summary, blissfully unaware of any thesis or purpose. They produce field notes, not arguments.


### Step 2 - Postdoc Bots
The postdocs are more ambitious. They receive the grad bots’ notes and a thesis statement. Their task is to rate each week’s relevance (0–100).

Scores >30/100: the week’s summary is sent to the professor.

Scores >80/100: the verbatim transcripts go to the professor (we do have a context window to worry about).

You might think this lab would never hit the context window with this process. You’d be wrong.

### Step 3 - Research Assistant Bots
Reads the thesis and finds the most thematically relevant *Buffy* episodes. Their scripts are added to the reading pile.

### Step 4 - Professor Bot

At last, the professor. Given the postdocs’ selections of grad student summaries and ver batim chats, as well as the the RA’s episodes, the professor composes an academic paper. It is authoritative, theoretical, and unbothered by the fact that its “sources” are two people texting about *Buffy*.

Why send both summaries and transcripts? It’s partly a context-window optimization. And partly because academic redundancy is a sacred institution.

## Turning It Into an Autonomatic Paper Factory

This would have been enough — but it still required a human to provide the thesis. So we removed the humans.

### Researcher Bot 
Before any lab kicks off, a researcher bot reads the chats and proposes abstract ideas and potential papers. Each abstract enters the full academic pipeline above.

### Peer Review Bot
When each paper generation process finishes, the paper is submitted to multiple eer reviewer bots for review. If rejected, it returns to the lab, where the professor must revise it using the reviewer’s comments.

This continues until all papers are accepted — at which point the conference is considered complete.

Of course this is where the simulation breaks down — in a human academic lab, the papers would actually be written and revised by the grad students. No professor necessary.