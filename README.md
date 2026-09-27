# Digital-Krishivani-LLM

An agricultural image analysis tool that uses Groq's LLaMA vision model to identify crop diseases and provide solutions in Hindi.

## Quick Start

### Prerequisites
- Python 3.9+
- Groq API key (get one at https://console.groq.com)

### Setup

1. Clone and install dependencies:
```bash
pip install langchain langchain-groq python-dotenv
```

2. Create `.env` file with your Groq API key:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Usage

Run image analysis from the command line:
```bash
python chat\ model.py apple-scab-1-460x263.webp
```

Or without arguments to be prompted:
```bash
python chat\ model.py
```

The tool will return a JSON response with:
- **problem**: What disease/issue was detected
- **reason**: Why it's happening
- **solution**: How to fix it
- **other_points**: Additional tips

## Project Structure

- `chat model.py` — CLI entry point
- `conveter.py` — Core analysis logic (image encoding, Groq API integration)
- `.env.example` — Template for environment configuration

## How It Works

1. Image is encoded as base64
2. Sent to Groq's LLaMA 4 Scout 17B vision model with Hindi prompt
3. Model response is parsed as JSON
4. Structured results returned to user
