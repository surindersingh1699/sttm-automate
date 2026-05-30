"""Run a local audio file through the headless pipeline and dump every
transcription + matcher decision. Useful for videos that aren't in the
OCR ground-truth dataset — gives raw visibility into what the model sees
and what the matcher does with it.

Usage:
    python scripts/diagnose_video.py wReYRaZBfH0
    python scripts/diagnose_video.py path/to/some.opus

Output: prints a chronological trace to stdout AND writes a JSONL of
every event to tests/eval/runs/diag_<video>_<runid>/events.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402
from src.transcription.factory import pin_indic_best_settings  # noqa: E402


def _wire():
    rt = json.loads((ROOT / ".runtime_settings.json").read_text(encoding="utf-8"))
    if (mid := rt.get("hf_model_id")) in config.whisper.available_models:
        config.whisper.apply_model_id(mid)
    config.whisper.engine = rt.get("engine", "indicconformer")
    if (prec := rt.get("onnx_precision")) in config.whisper.available_precisions:
        config.whisper.onnx_precision = prec
    config.whisper.lm_enabled = bool(rt.get("lm_enabled", False))
    for k, v in (rt.get("streaming") or {}).items():
        if hasattr(config.streaming, k):
            setattr(config.streaming, k, v)
    pin_indic_best_settings()
    # Force naive mode so this matches normal IndicConformer behaviour
    # regardless of what runtime_settings says (silero VAD would starve
    # the pipeline on kirtan).
    config.streaming.streaming_mode = "naive"


def _load_audio(path: Path) -> "numpy.ndarray":
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data.astype(np.float32)


async def _main(target: str, start_s: float, max_s: float | None):
    _wire()
    cache = ROOT / "tests" / "eval" / "cache" / "audio"
    if (cache / f"{target}.opus").exists():
        audio_path = cache / f"{target}.opus"
        video_id = target
    elif Path(target).exists():
        audio_path = Path(target)
        video_id = audio_path.stem
    else:
        print(f"Audio not found for '{target}' — checked {cache}/{target}.opus and {target}")
        sys.exit(1)

    print(f"[Diag] engine={config.whisper.engine} lm={config.whisper.lm_enabled} "
          f"mode={config.streaming.streaming_mode} precision={config.whisper.onnx_precision}")
    print(f"[Diag] Loading audio: {audio_path}")
    audio = _load_audio(audio_path)
    print(f"[Diag] Audio: {len(audio)/16000:.1f}s @ 16kHz")
    sr = 16000
    if start_s:
        audio = audio[int(start_s * sr):]
        print(f"[Diag] Skipped first {start_s:.0f}s — remaining {len(audio)/sr:.1f}s")
    if max_s is not None:
        audio = audio[: int(max_s * sr)]
        print(f"[Diag] Capped to {max_s:.0f}s — feeding {len(audio)/sr:.1f}s")

    run_id = uuid.uuid4().hex[:8]
    out_dir = ROOT / "tests" / "eval" / "runs" / f"diag_{video_id}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    f_events = open(events_path, "w")
    t0 = time.monotonic()
    print(f"[Diag] Writing events → {events_path}")
    print()

    async def broadcast(msg: dict):
        msg = {"t": round(time.monotonic() - t0, 3), **msg}
        f_events.write(json.dumps(msg, ensure_ascii=False) + "\n")
        f_events.flush()
        tp = msg.get("type")
        if tp == "transcription" and msg.get("text"):
            print(f"  [{msg['t']:6.1f}] ASR  rtf={msg.get('rtf',0):.2f}  '{msg['text'][:80]}'  fl={msg.get('first_letters','')[:30]}")
        elif tp == "shabad_locked":
            sh = msg.get("shabad", {})
            print(f"  [{msg['t']:6.1f}] LOCK shabad_id={msg.get('shabad_id')} score={sh.get('score','?')} action={sh.get('action','?')} "
                  f"word_overlap={sh.get('word_overlap','?')} gurmukhi='{sh.get('unicode','')[:60]}'")
        elif tp == "shabad_switched":
            print(f"  [{msg['t']:6.1f}] SWITCH new={msg.get('new_shabad_id')} old={msg.get('old_shabad_id')} reason={msg.get('reason','')}")
        elif tp == "force_unlock":
            print(f"  [{msg['t']:6.1f}] UNLOCK")
        elif tp == "candidate_set":
            cands = msg.get("candidates", [])[:3]
            if cands:
                desc = ", ".join(f"sid={c.get('shabad_id')}:{c.get('score','?')}" for c in cands)
                print(f"  [{msg['t']:6.1f}] CAND {desc}")

    from src.pipeline.orchestrator import PipelineOrchestrator  # noqa: PLC0415
    from tests.eval.mock_controller import MockSTTMController  # noqa: PLC0415

    controller = MockSTTMController()
    controller.reset()
    orch = PipelineOrchestrator(controller=controller, broadcast=broadcast, audio_device=-1)

    await asyncio.to_thread(orch.transcriber.load)
    await orch.controller.connect()
    orch._audio_source = "remote"
    orch.running = True

    tasks = [
        asyncio.create_task(orch._capture_tick_task()),
        asyncio.create_task(orch._decode_loop()),
    ]
    # Feed audio at 1× — same as headless eval driver
    samplerate = 16000
    chunk_samples = int(samplerate * 0.5)  # 500 ms slices
    fed = 0
    try:
        while fed < len(audio):
            slice_ = audio[fed : fed + chunk_samples]
            orch.audio.push_external(slice_)
            fed += len(slice_)
            await asyncio.sleep(len(slice_) / samplerate)
    finally:
        orch.running = False
        for t in tasks:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        await orch.stop()
        f_events.close()

    # Quick summary
    with open(events_path) as fh:
        events = [json.loads(l) for l in fh]
    locks = [e for e in events if e.get("type") in ("shabad_locked", "shabad_switched")]
    tx_total = sum(1 for e in events if e.get("type") == "transcription")
    tx_nonempty = sum(1 for e in events if e.get("type") == "transcription" and e.get("text","").strip())
    print()
    print(f"[Diag] === SUMMARY ===")
    print(f"  Audio:           {len(audio)/16000:.1f}s")
    print(f"  Transcriptions:  {tx_nonempty}/{tx_total} non-empty ({tx_nonempty/max(tx_total,1)*100:.0f}%)")
    print(f"  Lock events:     {len(locks)}")
    distinct = {l.get('shabad_id') or l.get('new_shabad_id') for l in locks}
    print(f"  Distinct shabads locked: {sorted(s for s in distinct if s)}")
    print(f"  Full event log:  {events_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("target", help="video_id (from cache) or audio path")
    p.add_argument("--start", type=float, default=0.0, help="skip first N seconds")
    p.add_argument("--max", dest="max_s", type=float, default=None,
                   help="cap audio at N seconds (default: full)")
    args = p.parse_args()
    asyncio.run(_main(args.target, args.start, args.max_s))
