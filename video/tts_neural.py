"""Neural narration via edge-tts -> audio/s{i}.wav (44.1 kHz mono).

    python tts_neural.py [voice] [rate]
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).parent
VOICE = sys.argv[1] if len(sys.argv) > 1 else "en-US-AndrewMultilingualNeural"
RATE = sys.argv[2] if len(sys.argv) > 2 else "+6%"
FF = os.environ.get("FFMPEG", "ffmpeg")


async def main():
    lines = json.loads((ROOT / "narration.json").read_text(encoding="utf-8"))
    (ROOT / "audio").mkdir(exist_ok=True)
    for i, ln in enumerate(lines):
        mp3 = ROOT / "audio" / f"s{i}.mp3"
        for attempt in range(5):
            try:
                await edge_tts.Communicate(ln["say"], VOICE, rate=RATE).save(str(mp3))
                break
            except Exception as e:  # transient service hiccups
                print("retry", i, attempt, type(e).__name__)
                await asyncio.sleep(2 + attempt * 2)
        else:
            raise SystemExit(f"TTS failed for line {i}")
        subprocess.run([FF, "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", "44100",
                        str(ROOT / "audio" / f"s{i}.wav")], check=True)
        print("ok", i)


asyncio.run(main())
