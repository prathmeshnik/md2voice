import os
import sys
import asyncio
import argparse
from pathlib import Path

import tts_pipeline as pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a single markdown file into voice-friendly audio"
    )
    parser.add_argument("file", help="Path to the .md file to process")
    parser.add_argument("--voice", default=pipe.DEFAULT_VOICE, help=f"Voice name (default: {pipe.DEFAULT_VOICE})")
    parser.add_argument("--model", default=pipe.DEFAULT_MODEL, help=f"Cerebras model (default: {pipe.DEFAULT_MODEL})")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true", help="Local TTS via Piper")
    group.add_argument("--online", action="store_true", help="Online TTS via edge-tts (default)")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--cpu", action="store_true", help="Use CPU for local TTS")
    device.add_argument("--gpu", action="store_true", help="Use GPU for local TTS (default)")
    return parser.parse_args()


async def main():
    args = parse_args()
    source_path = Path(args.file).resolve()

    if not source_path.exists():
        print(f"Error: File '{source_path}' does not exist")
        sys.exit(1)
    if not source_path.is_file():
        print(f"Error: '{source_path}' is not a file")
        sys.exit(1)
    if source_path.suffix.lower() != ".md":
        print(f"Error: '{source_path}' is not a .md file")
        sys.exit(1)

    print(f"File: {source_path}")

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: CEREBRAS_API_KEY not set")
        print("  $env:CEREBRAS_API_KEY = 'your-key-here'")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=pipe.CEREBRAS_BASE_URL)

    use_local_tts = args.local or not args.online
    use_cuda = not args.cpu if use_local_tts else False
    piper_voice = None

    if use_local_tts:
        piper_voice_name = args.voice if args.voice != pipe.DEFAULT_VOICE else pipe.DEFAULT_PIPER_VOICE
        device_label = "GPU" if use_cuda else "CPU"
        print(f"  TTS: Piper ({piper_voice_name}) on {device_label}")
        models_dir = Path(pipe.MODELS_DIR).resolve()
        piper_voice = pipe.load_piper_voice(models_dir, piper_voice_name, use_cuda=use_cuda)
    else:
        print(f"  TTS: edge-tts ({args.voice})")

    print(f"  Model: {args.model} (Cerebras)")

    content = pipe.read_file(source_path)
    if not content or len(content.strip()) < 10:
        print(f"File is too short or empty — nothing to process")
        return

    system_prompt = pipe.build_system_prompt(f"Single file: {source_path.name}", content[:500])
    print(f"\n=== Processing: {source_path.name}")

    print(f"  Calling Cerebras API ({args.model})...")
    voice_text = await pipe.call_cerebras(client, system_prompt, content, args.model)
    if not voice_text:
        print(f"  [SKIP] No response from Cerebras")
        return

    audio_output = source_path.with_suffix(".mp3" if piper_voice is None else ".wav")
    if piper_voice is not None:
        print(f"  Generating audio (Piper)...")
        pipe.synthesize_piper(piper_voice, voice_text, audio_output)
    else:
        print(f"  Generating audio ({args.voice})...")
        await pipe.synthesize_edge(voice_text, audio_output, args.voice)

    pipe.inject_audio_tag(source_path, audio_output)
    print(f"\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
