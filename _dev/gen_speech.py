import asyncio, os, subprocess
import edge_tts

DS = "/ai/VoiceClone/data/_smoke2"
OUT = "/ai/VoiceClone/outputs"
os.makedirs(DS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

TARGET_VOICE = "en-US-GuyNeural"      # voice to clone
INPUT_VOICE  = "en-US-AriaNeural"     # different voice -> file to transform

TARGET_TEXTS = [
    "The morning sun rose slowly over the quiet harbor as the fishermen prepared their boats.",
    "She opened the old wooden door and stepped into a room filled with dust and forgotten memories.",
    "Technology keeps changing the way we live, work, and communicate across great distances.",
    "He walked along the river path, listening to the birds and thinking about the day ahead.",
    "Learning a new language takes patience, practice, and a willingness to make mistakes.",
    "The scientists gathered around the table to discuss the surprising results of their experiment.",
]
INPUT_TEXT = "This is a test sentence that we will convert into the cloned target voice for verification."


async def synth(text, voice, path):
    await edge_tts.Communicate(text, voice).save(path)


def to_wav(src, dst, sr):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-y", "-i", src, "-ar", str(sr), "-ac", "1", dst], check=True)
    os.remove(src)


async def main():
    for i, t in enumerate(TARGET_TEXTS):
        mp3 = f"{DS}/t_{i}.mp3"
        await synth(t, TARGET_VOICE, mp3)
        to_wav(mp3, f"{DS}/t_{i}.wav", 40000)
    mp3 = f"{OUT}/_smoke2_input.mp3"
    await synth(INPUT_TEXT, INPUT_VOICE, mp3)
    to_wav(mp3, f"{OUT}/_smoke2_input.wav", 44100)
    print("GEN_DONE")


asyncio.run(main())
