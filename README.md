# my-llm-twin

Fine-tune an open-source LLM on your private messages so it texts like you.

The whole pipeline runs locally — your conversations never leave your machine. You export your DMs, parse them, build a training dataset, and LoRA fine-tune a model. That's it.

Currently supports **Facebook Messenger**. More soon.

## How it works

```
Export DMs (manual) -> Parse -> Build Dataset -> Train (LoRA) -> Chat (not yet implemented)
```

1. You export your Messenger data from Facebook as JSON
2. `parse` extracts 1:1 conversations, fixes encoding, filters by language
3. `build-dataset` segments conversations, builds sliding-window training examples
4. `train` LoRA fine-tunes Llama 3.1 8B Instruct on your data

## Requirements

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM and CUDA
- A Hugging Face account (Llama 3.1 is a gated model, you need to request access)

## Install

```bash
git clone <repo-url>
cd my-llm-twin
pip install -e .
```

## Setup

Run the interactive setup wizard:

```bash
my-llm-twin init
```

It will ask for your Messenger display name and target language, then create `config.yaml` for you.

All other settings (training hyperparameters, paths, etc.) use sensible defaults. Edit `config.yaml` directly if you need to tweak them.

## Export your DMs

1. Go to **facebook.com** > **Settings and privacy** > **Settings**
2. In **Account Center**, go to **Your information and permissions**
3. Click **Download your information** > **Export to device**
4. Select **Messages** only, pick **JSON** format, choose your time range
5. Hit **Start export** and wait for the email

Download the ZIP and drop it into `data/raw/` (no need to unzip).

More details in [docs/export-guide.md](docs/export-guide.md).

## Run the pipeline

### 1. Parse

```bash
my-llm-twin parse
```

Reads your exports from `data/raw/`, extracts 1:1 Messenger conversations, fixes Facebook's broken UTF-8 encoding, filters by language, and saves structured JSON files to `data/parsed/`.

### 2. Build dataset

```bash
my-llm-twin build-dataset
```

Takes parsed conversations, splits them into segments (based on silence gaps), and builds chat-formatted training examples with a sliding context window. Outputs `train.jsonl` and `val.jsonl` in `data/dataset/`.

### 3. Train

```bash
my-llm-twin train
```

LoRA fine-tunes the base model on your dataset. Saves the adapter and tokenizer to `models/my-twin/`. This is where you need the GPU.

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

All tests use synthetic data, no real conversations involved.

## Project status

This is a work in progress. Parsing, dataset building, and training are functional. Chat inference (`my-llm-twin chat`) is not implemented yet.

## License

MIT
