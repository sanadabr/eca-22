# Voice Chat Interface

A voice chat interface that uses OpenAI's APIs for speech-to-text, language processing, and text-to-speech conversion.

## Overview

Hold down the spacebar to talk with an AI assistant. Your voice will be:
1. Transcribed to text
2. Processed by GPT-4
3. Converted back to speech
4. Played through your speakers

The interface features an animated ASCII art display that changes colors based on the current state (standby, listening, processing, speaking).

## Setup

### Installation
1. Clone the repository:
```bash
git clone https://github.com/infamous-flu/HAI-Template.git
cd HAI-Template
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

The key must start with lowercase `sk-`. Optional overrides (defaults shown):
```
TRANSCRIBE_MODEL=gpt-4o-mini-transcribe
CHAT_MODEL=gpt-5-nano
TTS_MODEL=gpt-4o-mini-tts
TTS_VOICE=alloy
```

### Running the Application

1. Start the server (it serves both the API and the page):
```bash
python main.py
```

2. Open http://localhost:8000 in Chrome, Firefox or Safari and allow microphone access.
   Use `localhost`, not a `file://` path, or the browser blocks the microphone.
