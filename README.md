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

## Usage

### Extract from Stream Topic
```bash
python main.py extract-stream --stream "general" --topic "daily-checkin" --limit 500
```

### Extract from Private Conversation
```bash
python main.py extract-private --users "user1@example.com,user2@example.com" --limit 1000
```

## Options

- `--limit`: Maximum number of messages to extract (default: 1000)
- `--output`: Custom output filename (optional)

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
      "id": 12345,
      "content": "Hello world!",
      "sender": "John Doe",
      "timestamp": 1704110400,
      "stream_name": "general",
      "topic": "daily-checkin"
    }
  ]
}
```

## What Gets Extracted

- ✅ Text messages with full content (including markdown)
- ✅ User names and timestamps
- ✅ Stream/topic information (for stream messages)
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