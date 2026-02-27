#!/usr/bin/env python3
"""
Setup audio routing for STTM Automate.

After installing BlackHole (`brew install blackhole-2ch`), this script:
1. Verifies BlackHole is available
2. Guides you through creating a Multi-Output Device in Audio MIDI Setup
3. Tests audio capture from BlackHole

The Multi-Output Device sends audio to both your speakers AND BlackHole,
so you hear kirtan normally while our app captures the same audio.
"""

import subprocess
import sys


def check_blackhole():
    """Check if BlackHole is installed and visible to the system."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if "blackhole" in dev["name"].lower():
                print(f"  BlackHole found: [{i}] {dev['name']} ({dev['max_input_channels']}ch input)")
                return i
        print("  BlackHole NOT found in audio devices.")
        print("  Install it: brew install blackhole-2ch")
        print("  Then REBOOT your Mac.")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def print_setup_instructions():
    """Print instructions for creating the Multi-Output Device."""
    print("""
=== Multi-Output Device Setup ===

1. Open Audio MIDI Setup:
   - Press Cmd+Space, type "Audio MIDI Setup", press Enter
   - Or: open /Applications/Utilities/Audio MIDI Setup.app

2. Create Multi-Output Device:
   - Click the "+" button at bottom-left
   - Select "Create Multi-Output Device"

3. Configure it:
   - Check BOTH of these:
     [x] MacBook Air Speakers (or your external speakers)
     [x] BlackHole 2ch
   - Make sure "MacBook Air Speakers" is listed FIRST (drag to reorder)
   - This ensures speakers are the clock source

4. Set as system output:
   - Right-click the Multi-Output Device
   - Select "Use This Device For Sound Output"
   - OR: System Settings > Sound > Output > Multi-Output Device

Now all system audio goes to both speakers AND BlackHole.
Our app captures from BlackHole with perfect quality.

Press Enter when done...""")
    input()


def test_capture(device_index: int):
    """Test audio capture from BlackHole."""
    import numpy as np
    import sounddevice as sd

    print(f"\n=== Testing BlackHole Capture (device {device_index}) ===")
    print("Play some audio (YouTube kirtan, music, etc.)...")
    print("Recording 5 seconds...\n")

    samplerate = 16000
    duration = 5
    audio = sd.rec(
        int(samplerate * duration),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    audio = audio.flatten()

    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))

    print(f"  RMS:  {rms:.4f}")
    print(f"  Peak: {peak:.4f}")

    if rms > 0.01:
        print(f"  Audio captured successfully!")
        # Quick whisper test
        try:
            from faster_whisper import WhisperModel
            print("\n  Running quick Whisper transcription...")
            model = WhisperModel("base", device="cpu", compute_type="int8")

            # Normalize
            if peak > 0.01:
                gain = min(0.7 / peak, 15.0)
                audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

            segments, _ = model.transcribe(audio, language="pa", beam_size=5, vad_filter=True)
            texts = [s.text.strip() for s in segments if s.text.strip()]
            if texts:
                print(f"  Transcription: \"{' '.join(texts)}\"")
            else:
                print("  No speech detected (try playing kirtan with vocals)")
        except Exception as e:
            print(f"  Whisper test skipped: {e}")
    elif rms > 0.001:
        print("  Audio is very quiet. Make sure:")
        print("  - Multi-Output Device is set as system output")
        print("  - Audio is actually playing")
    else:
        print("  No audio captured! Check setup:")
        print("  - Is Multi-Output Device set as system output?")
        print("  - Does it include BlackHole 2ch?")

    return rms > 0.01


def main():
    print("=== STTM Automate Audio Setup ===\n")

    print("Step 1: Checking BlackHole...")
    bh_index = check_blackhole()

    if bh_index is None:
        print("\nBlackHole not found. Install and reboot first:")
        print("  brew install blackhole-2ch")
        print("  # Then reboot your Mac")
        sys.exit(1)

    print("\nStep 2: Multi-Output Device setup")
    print_setup_instructions()

    print("Step 3: Testing audio capture...")
    success = test_capture(bh_index)

    if success:
        print("\n=== Setup Complete! ===")
        print("Run the dashboard: python -m src.main dashboard")
        print("BlackHole will be auto-detected as the input device.")
    else:
        print("\n=== Setup needs attention ===")
        print("Audio capture didn't work. Review the Multi-Output Device setup above.")


if __name__ == "__main__":
    main()
