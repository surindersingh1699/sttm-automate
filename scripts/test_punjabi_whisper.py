#!/usr/bin/env python3
"""
Test Whisper models on Punjabi speech and Kirtan audio clips.
Compares: stock whisper-small vs fine-tuned whisper-small-punjabi (nayaniiii)

Downloads sample clips from YouTube, transcribes with both models,
and evaluates accuracy against known ground truth.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path

# Add static ffmpeg to PATH
FFMPEG_DIR = os.path.expanduser("~/.local/lib/python3.12/site-packages/static_ffmpeg/bin/linux")
os.environ["PATH"] = f"{FFMPEG_DIR}:{os.environ['PATH']}"

# ── Audio clip definitions ──────────────────────────────────────────────
# Each clip has: url, start_time, duration, type (speech/kirtan), ground_truth
# Ground truth = the expected Gurmukhi text (first line or key words)

CLIPS = [
    # === KIRTAN CLIPS ===
    {
        "name": "japji_sahib_kirtan",
        "url": "https://www.youtube.com/watch?v=ZKwfMQP1Dps",
        "start": 30,
        "duration": 15,
        "type": "kirtan",
        "ground_truth": "ਸਤਿਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ",
        "first_letters": "ਸਕਪ",
    },
    {
        "name": "dhan_guru_nanak",
        "url": "https://www.youtube.com/watch?v=L1_GF2qfHbU",
        "start": 60,
        "duration": 15,
        "type": "kirtan",
        "ground_truth": "ਧੰਨੁ ਗੁਰੂ ਨਾਨਕ",
        "first_letters": "ਧਗਨ",
    },
    {
        "name": "asa_di_vaar",
        "url": "https://www.youtube.com/watch?v=tN92YGTX2BY",
        "start": 45,
        "duration": 15,
        "type": "kirtan",
        "ground_truth": "ਬਲਿਹਾਰੀ ਗੁਰ ਆਪਣੇ",
        "first_letters": "ਬਗਆ",
    },
    {
        "name": "rehraas_kirtan",
        "url": "https://www.youtube.com/watch?v=XQFZ3qYLDJM",
        "start": 30,
        "duration": 15,
        "type": "kirtan",
        "ground_truth": "ਸੋ ਦਰੁ ਕੇਹਾ",
        "first_letters": "ਸਦਕ",
    },
    {
        "name": "anand_sahib_kirtan",
        "url": "https://www.youtube.com/watch?v=NKKFHFQ6HGc",
        "start": 30,
        "duration": 15,
        "type": "kirtan",
        "ground_truth": "ਅਨੰਦੁ ਭਇਆ ਮੇਰੀ ਮਾਏ",
        "first_letters": "ਅਭਮਮ",
    },
    # === PUNJABI SPEECH CLIPS ===
    {
        "name": "punjabi_speech_1",
        "url": "https://www.youtube.com/watch?v=mYx8_4LRbvg",
        "start": 30,
        "duration": 15,
        "type": "speech",
        "ground_truth": "",  # will evaluate qualitatively
        "first_letters": "",
    },
    {
        "name": "punjabi_news",
        "url": "https://www.youtube.com/watch?v=sKqN5g6cjNw",
        "start": 15,
        "duration": 15,
        "type": "speech",
        "ground_truth": "",
        "first_letters": "",
    },
    {
        "name": "gurbani_katha",
        "url": "https://www.youtube.com/watch?v=5RWQ5v7ePQE",
        "start": 60,
        "duration": 15,
        "type": "speech",
        "ground_truth": "",
        "first_letters": "",
    },
]

AUDIO_DIR = Path(__file__).parent.parent / "test_audio"


def download_clip(clip: dict) -> Path:
    """Download a YouTube clip segment as WAV."""
    AUDIO_DIR.mkdir(exist_ok=True)
    output_path = AUDIO_DIR / f"{clip['name']}.wav"

    if output_path.exists():
        print(f"  ✓ Already downloaded: {clip['name']}")
        return output_path

    print(f"  ↓ Downloading: {clip['name']} ({clip['type']})...")

    try:
        # Download with yt-dlp, extract audio segment
        temp_path = AUDIO_DIR / f"{clip['name']}_full.wav"

        # First download full audio
        result = subprocess.run(
            [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", "wav",
                "--output", str(temp_path),
                "--no-playlist",
                "--quiet",
                clip["url"],
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"  ✗ yt-dlp failed: {result.stderr[:200]}")
            return None

        # Find the actual downloaded file (yt-dlp may change extension)
        actual_temp = None
        for ext in [".wav", ".wav.wav"]:
            candidate = AUDIO_DIR / f"{clip['name']}_full{ext}"
            if candidate.exists():
                actual_temp = candidate
                break
        if not actual_temp:
            # Try glob
            for f in AUDIO_DIR.glob(f"{clip['name']}_full*"):
                actual_temp = f
                break

        if not actual_temp or not actual_temp.exists():
            print(f"  ✗ Downloaded file not found")
            return None

        # Extract segment with ffmpeg: convert to 16kHz mono WAV
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(actual_temp),
                "-ss", str(clip["start"]),
                "-t", str(clip["duration"]),
                "-ar", "16000",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Cleanup temp
        if actual_temp.exists():
            actual_temp.unlink()

        if result.returncode != 0:
            print(f"  ✗ ffmpeg failed: {result.stderr[:200]}")
            return None

        print(f"  ✓ Downloaded: {clip['name']} ({output_path.stat().st_size // 1024}KB)")
        return output_path

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout downloading {clip['name']}")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def load_audio_numpy(path: Path) -> np.ndarray:
    """Load WAV file as numpy array (float32, 16kHz mono)."""
    import wave
    with wave.open(str(path), 'rb') as wf:
        assert wf.getnchannels() == 1, "Expected mono"
        assert wf.getsampwidth() == 2, "Expected 16-bit"
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def transcribe_with_faster_whisper(audio_path: Path, model_size: str = "small") -> dict:
    """Transcribe using faster-whisper with given model size."""
    from faster_whisper import WhisperModel

    print(f"    Loading faster-whisper model: {model_size}...")
    start = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    load_time = time.time() - start

    print(f"    Transcribing...")
    start = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language="pa",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"threshold": 0.35},
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    transcribe_time = time.time() - start
    full_text = " ".join(text_parts)

    return {
        "text": full_text,
        "language": info.language,
        "language_prob": info.language_probability,
        "load_time": load_time,
        "transcribe_time": transcribe_time,
    }


def transcribe_with_hf_whisper(audio_path: Path, model_name: str) -> dict:
    """Transcribe using HuggingFace transformers whisper model."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch
    import torchaudio

    print(f"    Loading HF model: {model_name}...")
    start = time.time()
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    load_time = time.time() - start

    # Load audio
    audio = load_audio_numpy(audio_path)

    print(f"    Transcribing...")
    start = time.time()

    # Process in chunks of 30 seconds (Whisper's max)
    chunk_size = 16000 * 30  # 30 seconds at 16kHz
    text_parts = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        input_features = processor(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            # Force Punjabi language
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="punjabi", task="transcribe"
            )
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        text_parts.extend(transcription)

    transcribe_time = time.time() - start
    full_text = " ".join(t.strip() for t in text_parts if t.strip())

    return {
        "text": full_text,
        "language": "pa",
        "language_prob": None,
        "load_time": load_time,
        "transcribe_time": transcribe_time,
    }


def extract_first_letters(text: str) -> str:
    """Extract first Gurmukhi letter of each word."""
    # Gurmukhi Unicode range: 0x0A00 - 0x0A7F
    first_letters = []
    for word in text.split():
        # Find first Gurmukhi character
        for char in word:
            if '\u0A00' <= char <= '\u0A7F' and char not in 'ੁੂੋੌੇੈਿੀਾਂੰੱ਼':
                first_letters.append(char)
                break
    return "".join(first_letters)


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate character-level n-gram similarity (bigram containment)."""
    if not text1 or not text2:
        return 0.0

    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))

    bg1 = bigrams(text1.replace(" ", ""))
    bg2 = bigrams(text2.replace(" ", ""))

    if not bg1:
        return 0.0

    overlap = bg1 & bg2
    return len(overlap) / len(bg1)


def print_comparison(clip: dict, stock_result: dict, punjabi_result: dict):
    """Print side-by-side comparison."""
    print(f"\n{'='*70}")
    print(f"Clip: {clip['name']} ({clip['type']})")
    print(f"{'='*70}")

    if clip["ground_truth"]:
        print(f"Ground Truth:     {clip['ground_truth']}")
        print(f"Expected Letters: {clip['first_letters']}")

    print(f"\n--- Stock Whisper Small ---")
    print(f"  Text:     {stock_result['text'][:100]}")
    stock_letters = extract_first_letters(stock_result['text'])
    print(f"  Letters:  {stock_letters}")
    print(f"  Time:     {stock_result['transcribe_time']:.1f}s")

    if clip["ground_truth"]:
        sim = calculate_similarity(stock_result['text'], clip['ground_truth'])
        letter_sim = calculate_similarity(stock_letters, clip['first_letters'])
        print(f"  Text Sim: {sim:.0%}")
        print(f"  Letter Sim: {letter_sim:.0%}")

    print(f"\n--- Fine-tuned Whisper Punjabi ---")
    print(f"  Text:     {punjabi_result['text'][:100]}")
    punjabi_letters = extract_first_letters(punjabi_result['text'])
    print(f"  Letters:  {punjabi_letters}")
    print(f"  Time:     {punjabi_result['transcribe_time']:.1f}s")

    if clip["ground_truth"]:
        sim = calculate_similarity(punjabi_result['text'], clip['ground_truth'])
        letter_sim = calculate_similarity(punjabi_letters, clip['first_letters'])
        print(f"  Text Sim: {sim:.0%}")
        print(f"  Letter Sim: {letter_sim:.0%}")


def main():
    print("=" * 70)
    print("WHISPER PUNJABI MODEL COMPARISON TEST")
    print("Stock whisper-small vs nayaniiii/whisper-small-punjabi")
    print("=" * 70)

    # Step 1: Download clips
    print("\n📥 Downloading audio clips...\n")
    clip_paths = {}
    for clip in CLIPS:
        path = download_clip(clip)
        if path:
            clip_paths[clip["name"]] = path

    if not clip_paths:
        print("\n✗ No clips downloaded. Check your internet connection.")
        sys.exit(1)

    print(f"\n✓ Downloaded {len(clip_paths)}/{len(CLIPS)} clips\n")

    # Step 2: Transcribe with both models
    results = []

    for clip in CLIPS:
        if clip["name"] not in clip_paths:
            continue

        audio_path = clip_paths[clip["name"]]
        print(f"\n{'─'*50}")
        print(f"Processing: {clip['name']}")

        # Stock whisper-small
        print("  [1/2] Stock whisper-small...")
        try:
            stock_result = transcribe_with_faster_whisper(audio_path, "small")
        except Exception as e:
            print(f"  ✗ Stock model error: {e}")
            stock_result = {"text": "", "transcribe_time": 0, "load_time": 0}

        # Fine-tuned punjabi model
        print("  [2/2] Fine-tuned whisper-small-punjabi...")
        try:
            punjabi_result = transcribe_with_hf_whisper(
                audio_path, "nayaniiii/whisper-small-punjabi"
            )
        except Exception as e:
            print(f"  ✗ Punjabi model error: {e}")
            punjabi_result = {"text": "", "transcribe_time": 0, "load_time": 0}

        results.append({
            "clip": clip,
            "stock": stock_result,
            "punjabi": punjabi_result,
        })

    # Step 3: Print comparison
    print("\n\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    kirtan_stock_scores = []
    kirtan_punjabi_scores = []
    speech_stock_texts = []
    speech_punjabi_texts = []

    for r in results:
        print_comparison(r["clip"], r["stock"], r["punjabi"])

        if r["clip"]["ground_truth"]:
            stock_sim = calculate_similarity(r["stock"]["text"], r["clip"]["ground_truth"])
            punjabi_sim = calculate_similarity(r["punjabi"]["text"], r["clip"]["ground_truth"])
            kirtan_stock_scores.append(stock_sim)
            kirtan_punjabi_scores.append(punjabi_sim)

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if kirtan_stock_scores:
        print(f"\nKirtan Clips (text similarity to ground truth):")
        print(f"  Stock whisper-small:    avg {np.mean(kirtan_stock_scores):.0%}")
        print(f"  Fine-tuned punjabi:     avg {np.mean(kirtan_punjabi_scores):.0%}")

        if np.mean(kirtan_punjabi_scores) > np.mean(kirtan_stock_scores):
            improvement = np.mean(kirtan_punjabi_scores) - np.mean(kirtan_stock_scores)
            print(f"\n  → Fine-tuned model is {improvement:.0%} better on kirtan")
        else:
            diff = np.mean(kirtan_stock_scores) - np.mean(kirtan_punjabi_scores)
            print(f"\n  → Stock model is {diff:.0%} better (fine-tuned model didn't help)")

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print("=" * 70)

    if kirtan_stock_scores and kirtan_punjabi_scores:
        stock_avg = np.mean(kirtan_stock_scores)
        punjabi_avg = np.mean(kirtan_punjabi_scores)

        if punjabi_avg > stock_avg + 0.1:
            print("""
The fine-tuned Punjabi model significantly outperforms stock.
→ USE the fine-tuned model as your base
→ Further fine-tune on KIRTAN-SPECIFIC data for best results
→ Train on: Punjabi speech (Common Voice) + Kirtan recordings
""")
        elif abs(punjabi_avg - stock_avg) <= 0.1:
            print("""
Both models perform similarly on these clips.
→ The fine-tuning on Common Voice speech didn't help much for kirtan
→ You need KIRTAN-SPECIFIC training data
→ Train on: Kirtan recordings with known shabad labels
→ Use stock whisper-small as base (better supported)
""")
        else:
            print("""
Stock model performs better than the fine-tuned one.
→ The Punjabi fine-tuning may have overfit to speech patterns
→ Use stock whisper-small as your base
→ Fine-tune it yourself on KIRTAN data specifically
→ Don't mix too much regular speech — focus on singing/kirtan audio
""")

    # Save results to JSON
    results_path = AUDIO_DIR / "test_results.json"
    save_results = []
    for r in results:
        save_results.append({
            "clip": r["clip"]["name"],
            "type": r["clip"]["type"],
            "ground_truth": r["clip"]["ground_truth"],
            "stock_text": r["stock"]["text"],
            "stock_time": r["stock"]["transcribe_time"],
            "punjabi_text": r["punjabi"]["text"],
            "punjabi_time": r["punjabi"]["transcribe_time"],
        })
    with open(results_path, "w") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
