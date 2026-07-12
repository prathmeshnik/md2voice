# TTS Pipeline — Documentation → Voice

Converts markdown documentation into spoken audio with **independent LLM and TTS backends**.

| Component  | Online              | Local                        |
|------------|---------------------|------------------------------|
| **LLM**    | Cerebras API        | llama.cpp server (localhost) |
| **TTS**    | edge-tts (Microsoft)| Piper (GPU via onnxruntime)  |

Mix and match via `--llm` and `--tts` flags.

## How it works

```
source .md files
    │
    ├──→ LLM → voice-friendly text
    │    ├── online: Cerebras API
    │    └── local:  llama.cpp (http://localhost:8080/v1)
    │                └── saved to voice-notes/ (mirrored tree)
    │
    └──→ TTS engine → audio file
         ├── online: edge-tts → .mp3 (Microsoft cloud)
         └── local:  Piper    → .wav (GPU via onnxruntime)
```

## Setup

```powershell
# 1. Install deps (add onnxruntime-gpu for GPU local mode)
pip install edge-tts openai

# 2. Set Cerebras API key (only needed for --llm online)
$env:CEREBRAS_API_KEY = "your-key-here"
```

## Usage

```powershell
# Everything online (default)
python tts_pipeline.py --dir ./folder

# Fully local — llama.cpp LLM + Piper TTS
python tts_pipeline.py --dir ./folder --local

# Mixed: local LLM, online TTS
python tts_pipeline.py --dir ./folder --llm local --tts online

# Mixed: online LLM, local TTS
python tts_pipeline.py --dir ./folder --llm online --tts local
```

## Arguments

| Arg          | Default             | Description                                    |
|--------------|---------------------|------------------------------------------------|
| `--dir`      | `.`                 | Root directory to scan                         |
| `--voice`    | `en-US-JennyNeural` | edge-tts voice or Piper voice name             |
| `--model`    | `gpt-oss-120b`      | Cerebras model (ignored when `--llm local`)    |
| `--local`    | —                   | Shorthand: `--llm local --tts local`           |
| `--online`   | — (default)         | Shorthand: `--llm online --tts online`         |
| `--llm`      | `online`            | LLM backend: `online` (Cerebras) / `local` (llama.cpp) |
| `--tts`      | `online`            | TTS backend: `online` (edge-tts) / `local` (Piper)     |

## Single-file mode

```powershell
python tts_single.py doc.md                       # Cerebras + edge-tts
python tts_single.py doc.md --local               # llama.cpp + Piper
python tts_single.py doc.md --llm local --tts online  # llama.cpp + edge-tts
```

## Output

```
voice-notes/                          ← voice-friendly MDs (mirrored tree)
├── index.md
├── pipeline/
│   └── 01-parser.md
└── concepts/

project/pipeline/01-parser.mp3      ← online TTS (edge-tts)
project/pipeline/01-parser.wav      ← local TTS (Piper)

models/
└── en_US-lessac-medium.onnx          ← Piper model (auto-downloaded)
```

## Requirements

- Python 3.10+
- **Online LLM**: `CEREBRAS_API_KEY` environment variable
- **Local LLM**: llama.cpp server running at `http://localhost:8080/v1`
- **Local TTS**: NVIDIA GPU + `onnxruntime-gpu` for acceleration
