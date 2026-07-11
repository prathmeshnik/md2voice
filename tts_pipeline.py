import os
import sys
import time
import asyncio
import argparse
import wave
from pathlib import Path

import numpy as np


VOICE_NOTES_DIR = "voice-notes"
MODELS_DIR = "models"
DEFAULT_VOICE = "en-US-JennyNeural"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"
DEFAULT_MODEL = "gpt-oss-120b"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert markdown documentation into voice-friendly audio"
    )
    parser.add_argument(
        "--dir", default=".", help="Root directory to process (default: current dir)"
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Online: edge-tts voice. Local: Piper voice like en_US-lessac-medium (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Cerebras model (default: {DEFAULT_MODEL})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true", help="Local TTS via Piper")
    group.add_argument(
        "--online", action="store_true", help="Online TTS via edge-tts (default)"
    )
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--cpu", action="store_true", help="Use CPU for local TTS")
    device.add_argument("--gpu", action="store_true", help="Use GPU for local TTS (default)")
    return parser.parse_args()


def scan_directory(root: Path) -> dict:
    structure = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d != VOICE_NOTES_DIR
        ]
        md_files = sorted(f for f in filenames if f.endswith(".md"))
        if md_files:
            structure[str(Path(dirpath).relative_to(root))] = md_files
    return structure


def build_structure_text(root: Path, structure: dict) -> str:
    lines = [f"Project: {root.name}/"]
    for rel_path in sorted(structure, key=lambda x: (x != ".", x)):
        label = "." if rel_path == "." else rel_path
        lines.append(f"  {label}/")
        for f in structure[rel_path]:
            lines.append(f"    {f}")
    return "\n".join(lines)


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [ERROR] Could not read {path}: {e}")
        return ""


def build_system_prompt(structure_text: str, index_content: str) -> str:
    return f"""You are a technical documentation voice-over writer. Rewrite documentation into conversational, spoken-word text optimized for text-to-speech.

PROJECT STRUCTURE:
{structure_text}

PROJECT OVERVIEW (from index.md):
{index_content if index_content else "No index.md found — this is the project root."}

RULES:
- Write in natural, conversational English — as if explaining to someone verbally
- Use complete sentences. NO bullet points, NO tables, NO code fences, NO markdown formatting
- Include verbal signposts: "First...", "Next...", "For example...", "Let me explain...", "Here's how it works..."
- Keep all technical accuracy but explain concepts conversationally
- Output ONLY the voice-ready text. No preamble, no commentary, no markdown."""


def save_voice_md(text: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"  [SAVED] Voice MD -> {output_path}")


def get_files_in_order(root: Path, structure: dict) -> list:
    # Prioritize pipeline/ before concepts/ for logical continuity
    def folder_key(x):
        if x == ".":
            return (0, "")
        if x == "pipeline":
            return (1, x)
        return (2, x)

    index_entry = None
    rest = []
    for rel_path in sorted(structure, key=folder_key):
        for f in structure[rel_path]:
            fp = root / (Path(rel_path) / f if rel_path != "." else f)
            if rel_path == "." and f == "index.md":
                index_entry = fp
            else:
                rest.append(fp)
    return ([index_entry] if index_entry else []) + rest


# ── LLM (same for both modes) ─────────────────────────────

# Rate limiter: 5 calls per 60 seconds (sliding window)
_rate_calls: list[float] = []


async def _acquire():
    global _rate_calls
    now = time.monotonic()
    _rate_calls = [t for t in _rate_calls if now - t < 60]
    if len(_rate_calls) >= 5:
        wait = 60 - (now - _rate_calls[0])
        if wait > 0:
            print(f"  [RATE] Waiting {wait:.1f}s for rate limit (5 req/min)...")
            await asyncio.sleep(wait)
    _rate_calls.append(time.monotonic())


async def call_cerebras(client, system_prompt: str, file_content: str, model: str) -> str:
    await _acquire()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Rewrite this documentation file into voice-friendly text:\n\n{file_content}",
                },
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] Cerebras API call failed: {e}")
        return ""


# ── Online TTS (edge-tts) ─────────────────────────────────


async def synthesize_edge(text: str, output_path: Path, voice: str):
    import edge_tts

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        print(f"  [AUDIO] Saved -> {output_path}")
    except Exception as e:
        print(f"  [ERROR] edge-tts failed: {e}")


# ── Local TTS (Piper, GPU via onnxruntime) ─────────────────


def piper_download_url(voice: str) -> tuple:
    quality = voice.rsplit("-", 1)[-1]
    stem = voice.rsplit("-", 1)[0]
    lang_code = stem.split("-")[0]
    name = "-".join(stem.split("-")[1:])
    lang = lang_code.split("_")[0]
    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang_code}/{name}/{quality}/{voice}"
    return base + ".onnx", base + ".onnx.json"


def download_piper_model(models_dir: Path, voice: str):
    model_onnx = models_dir / f"{voice}.onnx"
    model_json = models_dir / f"{voice}.onnx.json"
    if model_onnx.exists() and model_json.exists():
        return model_onnx, model_json

    print(f"  Downloading Piper voice ({voice}) to {models_dir}...")
    models_dir.mkdir(parents=True, exist_ok=True)

    url_onnx, url_json = piper_download_url(voice)
    import urllib.request

    urllib.request.urlretrieve(url_onnx, str(model_onnx))
    urllib.request.urlretrieve(url_json, str(model_json))
    print(f"  Download complete.")
    return model_onnx, model_json


def load_piper_voice(models_dir: Path, voice: str, use_cuda: bool = True):
    from piper import PiperVoice

    model_onnx, model_json = download_piper_model(models_dir, voice)
    print(f"  Loading Piper voice (use_cuda={use_cuda})...")
    return PiperVoice.load(
        model_path=str(model_onnx), config_path=str(model_json), use_cuda=use_cuda
    )


def synthesize_piper(voice, text: str, output_path: Path):
    chunks = list(voice.synthesize(text))

    audio_parts = []
    sample_rate = None
    for chunk in chunks:
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        audio_parts.append(chunk.audio_int16_array)

    audio = np.concatenate(audio_parts)

    output_path = output_path.with_suffix(".wav")
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())

    print(f"  [AUDIO] Saved -> {output_path}")


# ── Audio embed injection ─────────────────────────────────


def get_audio_duration(path: Path) -> str:
    from mutagen import File
    try:
        audio = File(str(path))
        seconds = round(audio.info.length)
        m, s = divmod(seconds, 60)
        return f"{m} min {s} sec" if m else f"{s} sec"
    except Exception:
        return "unknown duration"


AUDIO_HEADER_MARKER = "**Audio version**"


def inject_audio_tag(source_path: Path, audio_path: Path):
    content = source_path.read_text(encoding="utf-8")
    audio_name = audio_path.name
    duration = get_audio_duration(audio_path)
    tag = (
        f"**Audio version** — Estimated duration: {duration}\n\n"
        f'<audio controls src="{audio_name}">\n'
        f"  Your browser does not support the audio element.\n"
        f"</audio>\n\n"
    )
    if AUDIO_HEADER_MARKER in content:
        return  # already injected
    source_path.write_text(tag + content, encoding="utf-8")
    print(f"  [EMBED] Audio tag -> {source_path}")


# ── Pipeline ──────────────────────────────────────────────


async def process_file(
    client,
    source_path: Path,
    root: Path,
    system_prompt: str,
    voice: str,
    model: str,
    piper_voice,
    next_file: Path = None,
):
    rel_path = source_path.relative_to(root)
    print(f"\n=== Processing: {rel_path}")

    content = read_file(source_path)
    if not content or len(content.strip()) < 10:
        print(f"  [SKIP] File too short or empty")
        return

    file_prompt = system_prompt
    if source_path.name == "index.md":
        extra = "\n\nINDEX FILE — SPECIAL RULES:\n- Keep to ~200 words (about 1-2 minutes when spoken aloud)"
        if next_file:
            try:
                rel = next_file.relative_to(root)
                extra += f"\n- End with a sentence suggesting the reader continue to '{rel}' next for the detailed breakdown"
            except ValueError:
                pass
        file_prompt = system_prompt + extra

    print(f"  Calling Cerebras API ({model})...")
    voice_text = await call_cerebras(client, file_prompt, content, model)
    if not voice_text:
        print(f"  [SKIP] No response from Cerebras")
        return

    save_voice_md(voice_text, root / VOICE_NOTES_DIR / rel_path)

    audio_output = source_path.with_suffix(".mp3" if piper_voice is None else ".wav")
    if piper_voice is not None:
        print(f"  Generating audio (Piper GPU)...")
        synthesize_piper(piper_voice, voice_text, audio_output)
    else:
        print(f"  Generating audio ({voice})...")
        await synthesize_edge(voice_text, audio_output, voice)

    inject_audio_tag(source_path, audio_output)


async def main():
    args = parse_args()
    root = Path(args.dir).resolve()

    if not root.exists():
        print(f"Error: Directory '{root}' does not exist")
        sys.exit(1)
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory")
        sys.exit(1)

    print(f"Scanning: {root}")
    structure = scan_directory(root)
    if not structure:
        print("No .md files found — nothing to process")
        return

    structure_text = build_structure_text(root, structure)
    total_files = sum(len(files) for files in structure.values())
    print(f"Found {total_files} .md file(s) across {len(structure)} folder(s)")
    print(f"\nDirectory structure:\n{structure_text}")

    index_content = (
        read_file(root / "index.md") if (root / "index.md").exists() else ""
    )
    system_prompt = build_system_prompt(structure_text, index_content)
    files = get_files_in_order(root, structure)

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: CEREBRAS_API_KEY not set")
        print("  $env:CEREBRAS_API_KEY = 'your-key-here'")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=CEREBRAS_BASE_URL)

    use_local_tts = args.local or not args.online
    use_cuda = not args.cpu if use_local_tts else False
    piper_voice = None

    if use_local_tts:
        piper_voice_name = args.voice if args.voice != DEFAULT_VOICE else DEFAULT_PIPER_VOICE
        device_label = "GPU" if use_cuda else "CPU"
        print(f"\n── Local TTS mode ({device_label}) ──")
        print(f"  LLM: {args.model} (Cerebras)")
        print(f"  TTS: Piper ({piper_voice_name}) on {device_label}")
        models_dir = Path(MODELS_DIR).resolve()
        piper_voice = load_piper_voice(models_dir, piper_voice_name, use_cuda=use_cuda)
    else:
        print(f"\n── Online mode ──")
        print(f"  LLM: {args.model} (Cerebras)")
        print(f"  TTS: edge-tts ({args.voice})")

    next_file_map = {}
    for i, fp in enumerate(files):
        if i + 1 < len(files):
            next_file_map[fp] = files[i + 1]

    print(f"\nProcessing {len(files)} file(s)...")
    for source_path in files:
        await process_file(
            client,
            source_path,
            root,
            system_prompt,
            args.voice,
            args.model,
            piper_voice,
            next_file_map.get(source_path),
        )

    print(f"\nDone!")
    print(f"Voice-friendly MDs saved in: {root / VOICE_NOTES_DIR / ''}")


if __name__ == "__main__":
    asyncio.run(main())
