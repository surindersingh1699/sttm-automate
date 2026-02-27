"""
Phase 1 Validation: Test faster-whisper with Punjabi audio.

Run: python scripts/test_whisper.py

Tests:
1. Model loading (tiny, base)
2. Punjabi transcription from a test recording
3. Output script detection (Gurmukhi vs Devanagari vs Roman)
4. VAD filter effectiveness
"""

import sys
import time
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)


def detect_script(text: str) -> str:
    """Detect what script the text is in."""
    for char in text:
        cp = ord(char)
        if 0x0A00 <= cp <= 0x0A7F:
            return "Gurmukhi"
        elif 0x0900 <= cp <= 0x097F:
            return "Devanagari"
        elif 0x0600 <= cp <= 0x06FF:
            return "Arabic/Shahmukhi"
        elif char.isascii() and char.isalpha():
            continue
    return "Roman/ASCII"


def test_model_loading():
    """Test loading different model sizes."""
    print("1. Testing model loading...")
    models_to_test = ["tiny", "base"]

    for model_name in models_to_test:
        print(f"\n   Loading '{model_name}' model (this downloads on first run)...")
        start = time.time()
        try:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            elapsed = time.time() - start
            print(f"   '{model_name}' loaded in {elapsed:.1f}s")
        except Exception as e:
            print(f"   ERROR loading '{model_name}': {e}")


def record_and_transcribe(duration: float = 5.0, model_size: str = "base"):
    """Record audio from mic and transcribe it."""
    samplerate = 16000

    print(f"\n2. Recording {duration}s of audio from your microphone...")
    print("   Speak or sing in Punjabi now!")

    try:
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()
        print(f"   Recorded {len(audio)} samples ({len(audio)/samplerate:.1f}s)")

        # Check if audio has signal
        rms = np.sqrt(np.mean(audio**2))
        print(f"   Audio RMS level: {rms:.4f} {'(very quiet - check mic!)' if rms < 0.01 else '(OK)'}")

    except Exception as e:
        print(f"   ERROR recording: {e}")
        print("   Available audio devices:")
        print(sd.query_devices())
        return

    # Transcribe
    print(f"\n3. Transcribing with '{model_size}' model (language=pa)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Without VAD
    start = time.time()
    segments, info = model.transcribe(
        audio,
        language="pa",
        beam_size=5,
        vad_filter=False,
    )
    segments_list = list(segments)
    elapsed = time.time() - start

    print(f"   Detected language: {info.language} (prob: {info.language_probability:.2f})")
    print(f"   Transcription time: {elapsed:.1f}s")
    print(f"   Segments: {len(segments_list)}")

    for i, seg in enumerate(segments_list):
        script = detect_script(seg.text)
        print(f"\n   Segment {i+1} [{seg.start:.1f}s - {seg.end:.1f}s]:")
        print(f"     Text: {seg.text}")
        print(f"     Script: {script}")

    # With VAD
    print(f"\n4. Transcribing with VAD filter...")
    start = time.time()
    segments_vad, info_vad = model.transcribe(
        audio,
        language="pa",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )
    segments_vad_list = list(segments_vad)
    elapsed = time.time() - start

    print(f"   With VAD: {len(segments_vad_list)} segments in {elapsed:.1f}s")
    for i, seg in enumerate(segments_vad_list):
        script = detect_script(seg.text)
        print(f"   Segment {i+1}: {seg.text} ({script})")


def list_audio_devices():
    """Show available audio input devices."""
    print("\n5. Available audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <-- DEFAULT" if i == sd.default.device[0] else ""
            print(f"   [{i}] {dev['name']} (inputs: {dev['max_input_channels']}){marker}")


def main():
    print("STTM Automate - Whisper Validation")
    print("=" * 60)

    list_audio_devices()

    print("\n" + "=" * 60)
    test_model_loading()

    print("\n" + "=" * 60)
    print("Ready to record. Make sure your microphone is working.")
    input("Press Enter to start recording (5 seconds)...")

    record_and_transcribe(duration=5.0, model_size="base")

    print("\n\nDone! Review the output to understand:")
    print("  - What script Whisper outputs for Punjabi (Gurmukhi/Devanagari/Roman)")
    print("  - How accurate the transcription is")
    print("  - Whether VAD filter helps")


if __name__ == "__main__":
    main()
