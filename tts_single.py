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
    parser.add_argument("--model", default=pipe.DEFAULT_MODEL, help=f"Cerebras model (default: {pipe.DEFAULT_MODEL}) — ignored when --llm local")
    parser.add_argument("--local", action="store_true", help="Shorthand: use local LLM + local TTS")
    parser.add_argument("--online", action="store_true", help="Shorthand: use online LLM + online TTS (default)")
    parser.add_argument("--llm", choices=["online", "local"], help="LLM backend (online: Cerebras, local: llama.cpp)")
    parser.add_argument("--tts", choices=["online", "local"], help="TTS backend (online: edge-tts, local: Piper)")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--cpu", action="store_true", help="Use CPU for local TTS")
    device.add_argument("--gpu", action="store_true", help="Use GPU for local TTS (default)")
    return parser.parse_args()


async def main():
    args = parse_args()

    llm_mode = "online"
    tts_mode = "online"
    if args.local:
        llm_mode = "local"
        tts_mode = "local"
    if args.online:
        llm_mode = "online"
        tts_mode = "online"
    if args.llm is not None:
        llm_mode = args.llm
    if args.tts is not None:
        tts_mode = args.tts

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

    if llm_mode == "online":
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            print("Error: CEREBRAS_API_KEY not set")
            print("  $env:CEREBRAS_API_KEY = 'your-key-here'")
            sys.exit(1)
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=pipe.CEREBRAS_BASE_URL)
    else:
        from openai import OpenAI
        client = OpenAI(api_key="sk-no-key-required", base_url=pipe.LLAMA_CPP_BASE_URL)

    use_cuda = not args.cpu if tts_mode == "local" else False
    piper_voice = None

    llm_label = "llama.cpp" if llm_mode == "local" else f"{args.model} (Cerebras)"
    if tts_mode == "local":
        piper_voice_name = args.voice if args.voice != pipe.DEFAULT_VOICE else pipe.DEFAULT_PIPER_VOICE
        device_label = "GPU" if use_cuda else "CPU"
        print(f"  TTS: Piper ({piper_voice_name}) on {device_label}")
        models_dir = Path(pipe.MODELS_DIR).resolve()
        piper_voice = pipe.load_piper_voice(models_dir, piper_voice_name, use_cuda=use_cuda)
    else:
        print(f"  TTS: edge-tts ({args.voice})")

    print(f"  LLM: {llm_label}")

    content = pipe.read_file(source_path)
    if not content or len(content.strip()) < 10:
        print(f"File is too short or empty — nothing to process")
        return

    system_prompt = pipe.build_system_prompt(f"Single file: {source_path.name}", content[:500])
    print(f"\n=== Processing: {source_path.name}")

    if llm_mode == "local":
        print(f"  Calling llama.cpp ({args.model})...")
        voice_text = await pipe.call_llama_cpp(client, system_prompt, content, args.model)
    else:
        print(f"  Calling Cerebras API ({args.model})...")
        voice_text = await pipe.call_cerebras(client, system_prompt, content, args.model)
    if not voice_text:
        print(f"  [SKIP] No response from LLM")
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
