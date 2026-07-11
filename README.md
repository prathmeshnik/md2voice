# TTS Pipeline — Documentation → Voice

Converts markdown documentation into spoken audio using **Cerebras AI** for voice-friendly rewriting, with **two TTS modes**.

## How it works

```
source .md files
    │
    ├──→ Cerebras API → voice-friendly text
    │                     └── saved to voice-notes/ (mirrored tree)
    │
    └──→ TTS engine → audio file
         ├── online:  edge-tts → .mp3 (Microsoft cloud)
         └── local:   Piper    → .wav (GPU via onnxruntime)
```

## Setup

```powershell
# 1. Install deps (add onnxruntime-gpu for GPU local mode)
pip install edge-tts openai

# 2. Set Cerebras API key
$env:CEREBRAS_API_KEY = "your-key-here"
```

## Usage

```powershell
# Online mode (default) — edge-tts via Microsoft cloud
python tts_pipeline.py --dir ./folder

# Local TTS mode — Piper runs on GPU (model auto-downloads to models/)
python tts_pipeline.py --dir ./folder --local
```

## Arguments

| Arg        | Default                    | Description                          |
|------------|----------------------------|--------------------------------------|
| `--dir`    | `.`                        | Root directory to scan               |
| `--voice`  | `en-US-JennyNeural`        | edge-tts voice name                  |
| `--model`  | `gpt-oss-120b`             | Cerebras model name                  |
| `--local`  | —                          | Use local TTS (Piper, GPU)           |
| `--online` | — (default)                | Use online TTS (edge-tts)            |

## Output

```
voice-notes/                          ← voice-friendly MDs (mirrored tree)
├── index.md
├── pipeline/
│   └── 01-parser.md
└── concepts/

project/pipeline/01-parser.mp3      ← online mode (edge-tts)
project/pipeline/01-parser.wav      ← local mode (Piper)

models/
└── en_US-lessac-medium.onnx          ← Piper model (auto-downloaded)
```

## Requirements

- Python 3.10+
- Internet for Cerebras API (required in both modes)
- `CEREBRAS_API_KEY` environment variable
- **Local TTS**: NVIDIA GPU + `onnxruntime-gpu` for acceleration
