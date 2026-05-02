"""Benchmark surt-small-v3 across engines/quantizations on cached kirtan audio.

Decodes a fixed set of 10s windows (the pipeline's default) through every
engine variant and reports:

  - on-disk size
  - total decode time + RTF
  - first-letter-code overlap with the f16 ggml reference

The first-letter codes are what the matcher actually consumes, so FL-overlap
is a far better proxy for downstream lock accuracy than raw WER.

Variants compared (skipped automatically if backend not installed):

  faster-whisper-int8    data/surt-small-v3-ct2/
  mlx-4bit               data/surt-small-v3-mlx/
  whisper-cpp-f16        data/surt-small-v3.ggml
  whisper-cpp-q8_0       data/surt-small-v3-q8_0.ggml
  whisper-cpp-q5_1       data/surt-small-v3-q5_1.ggml
  whisper-cpp-q4_0       data/surt-small-v3-q4_0.ggml
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.transcription.transliterate import extract_first_letters, gurmukhi_to_ascii, normalize_for_fullword_search  # noqa
from src.matcher.scorer import _strip_matras, _char_4grams, _ngram_overlap  # noqa


AUDIO = ROOT / "tests/eval/cache/audio/-Dyi8-Qyx4I.opus"
WINDOW_S = 10.0
NUM_WINDOWS = 12
START_OFFSET_S = 30.0   # skip intro
STRIDE_S = 25.0


def load_audio_windows() -> list[np.ndarray]:
    """Decode the cached opus to mono 16k float32 and slice into windows."""
    import subprocess

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(AUDIO),
        "-ac", "1", "-ar", "16000",
        "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    pcm = np.frombuffer(raw, dtype=np.float32)
    sr = 16000
    win = int(WINDOW_S * sr)
    out = []
    for i in range(NUM_WINDOWS):
        start = int((START_OFFSET_S + i * STRIDE_S) * sr)
        end = start + win
        if end > len(pcm):
            break
        out.append(pcm[start:end].copy())
    return out


def fl_codes(text: str) -> str:
    return gurmukhi_to_ascii(extract_first_letters(text or ""))


def fl_overlap(a: str, b: str) -> float:
    """Char-level F1 between two first-letter strings."""
    if not a or not b:
        return 0.0
    sa, sb = list(a), list(b)
    common = 0
    sb_copy = sb.copy()
    for c in sa:
        if c in sb_copy:
            sb_copy.remove(c)
            common += 1
    if common == 0:
        return 0.0
    p = common / len(sa)
    r = common / len(b)
    return 2 * p * r / (p + r)


def word_overlap_f1(a: str, b: str) -> float:
    """Matra-stripped Gurmukhi word-set F1 — what the line matcher actually uses."""
    if not a or not b:
        return 0.0
    qa = set(_strip_matras(normalize_for_fullword_search(a)).split())
    qb = set(_strip_matras(normalize_for_fullword_search(b)).split())
    if not qa or not qb:
        return 0.0
    common = qa & qb
    if not common:
        return 0.0
    p = len(common) / len(qa)
    r = len(common) / len(qb)
    return 2 * p * r / (p + r)


def ngram_overlap_score(a: str, b: str) -> float:
    """Char-4-gram overlap-coefficient on normalized Gurmukhi — used by ngram_line_scoring."""
    if not a or not b:
        return 0.0
    na = normalize_for_fullword_search(a)
    nb = normalize_for_fullword_search(b)
    if not na or not nb:
        return 0.0
    return _ngram_overlap(_char_4grams(na), _char_4grams(nb))


@dataclass
class VariantResult:
    name: str
    size_mb: float
    total_s: float = 0.0
    audio_s: float = 0.0
    texts: list[str] = field(default_factory=list)
    fls: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def rtf(self) -> float:
        return self.total_s / self.audio_s if self.audio_s else 0.0


def file_size_mb(p: Path) -> float:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return p.stat().st_size / 1e6


# ── faster-whisper (CT2 int8) ────────────────────────────────────────────────

def make_faster_whisper(model_dir: Path) -> Callable[[np.ndarray], str]:
    from faster_whisper import WhisperModel
    m = WhisperModel(str(model_dir), device="cpu", compute_type="int8")

    def transcribe(pcm: np.ndarray) -> str:
        segs, _ = m.transcribe(
            pcm, language="pa", beam_size=5, temperature=0.0,
            condition_on_previous_text=False, vad_filter=False,
        )
        return " ".join(s.text for s in segs).strip()

    return transcribe


# ── mlx-whisper ──────────────────────────────────────────────────────────────

def make_mlx_whisper(model_dir: Path) -> Callable[[np.ndarray], str]:
    import mlx_whisper

    def transcribe(pcm: np.ndarray) -> str:
        out = mlx_whisper.transcribe(
            pcm, path_or_hf_repo=str(model_dir),
            language="pa", temperature=0.0, condition_on_previous_text=False,
        )
        return (out.get("text") or "").strip()

    return transcribe


# ── whisper.cpp via pywhispercpp ─────────────────────────────────────────────

def make_whisper_cpp(ggml_path: Path) -> Callable[[np.ndarray], str]:
    from pywhispercpp.model import Model
    m = Model(model=str(ggml_path), n_threads=4, language="pa", print_progress=False, print_realtime=False)

    def transcribe(pcm: np.ndarray) -> str:
        segs = m.transcribe(pcm, language="pa")
        return " ".join((s.text for s in segs)).strip()

    return transcribe


VARIANTS: list[tuple[str, Path, Callable[[Path], Callable[[np.ndarray], str]]]] = [
    ("whisper-cpp-f16",  ROOT / "data/surt-small-v3.ggml",       make_whisper_cpp),
    ("whisper-cpp-q8_0", ROOT / "data/surt-small-v3-q8_0.ggml",  make_whisper_cpp),
    ("whisper-cpp-q5_1", ROOT / "data/surt-small-v3-q5_1.ggml",  make_whisper_cpp),
    ("whisper-cpp-q4_0", ROOT / "data/surt-small-v3-q4_0.ggml",  make_whisper_cpp),
    ("faster-whisper-int8", ROOT / "data/surt-small-v3-ct2",     make_faster_whisper),
    ("mlx-4bit",         ROOT / "data/surt-small-v3-mlx",        make_mlx_whisper),
]


def run() -> None:
    print(f"[bench] loading {NUM_WINDOWS}× {WINDOW_S}s audio windows from {AUDIO.name}")
    windows = load_audio_windows()
    audio_s = len(windows) * WINDOW_S
    print(f"[bench] decoded {len(windows)} windows = {audio_s:.0f}s of audio\n")

    results: list[VariantResult] = []
    for name, path, factory in VARIANTS:
        if not path.exists():
            print(f"[skip] {name}: missing {path}")
            continue

        size = file_size_mb(path)
        res = VariantResult(name=name, size_mb=round(size, 1), audio_s=audio_s)
        print(f"\n=== {name}  ({size:.1f} MB) ===")
        try:
            transcribe = factory(path)
        except Exception as e:
            res.error = f"load: {e}"
            print(f"  [error] {e}")
            results.append(res)
            continue

        # warm-up
        try:
            _ = transcribe(windows[0])
        except Exception as e:
            res.error = f"warmup: {e}"
            print(f"  [error] warmup {e}")
            results.append(res)
            continue

        t0 = time.monotonic()
        for i, w in enumerate(windows):
            text = transcribe(w)
            res.texts.append(text)
            res.fls.append(fl_codes(text))
        res.total_s = round(time.monotonic() - t0, 2)
        rtf = res.total_s / audio_s
        print(f"  decoded {len(windows)} windows in {res.total_s}s  (RTF={rtf:.3f})")
        for i, (t, fl) in enumerate(zip(res.texts, res.fls)):
            preview = (t[:60] + "…") if len(t) > 60 else t
            print(f"   [{i:02d}] fl={fl[:24]:<24}  {preview}")
        results.append(res)

    # --- comparison vs whisper-cpp-f16 reference ---
    ref = next((r for r in results if r.name == "whisper-cpp-f16" and r.fls), None)
    print("\n\n========  SUMMARY  ========")
    header = f"{'engine':<22}{'size_MB':>9}{'rtf':>8}{'fl_F1':>9}{'word_F1':>10}{'ngram':>9}"
    print(header)
    print("-" * len(header))
    summary_rows = []
    for r in results:
        if r.error:
            print(f"{r.name:<22}  ERROR: {r.error}")
            summary_rows.append({"engine": r.name, "error": r.error, "size_mb": r.size_mb})
            continue
        if ref and r.fls:
            fl_ov = sum(fl_overlap(a, b) for a, b in zip(r.fls, ref.fls)) / len(r.fls)
            word_ov = sum(word_overlap_f1(a, b) for a, b in zip(r.texts, ref.texts)) / len(r.texts)
            ngram_ov = sum(ngram_overlap_score(a, b) for a, b in zip(r.texts, ref.texts)) / len(r.texts)
        else:
            fl_ov = word_ov = ngram_ov = float("nan")
        print(f"{r.name:<22}{r.size_mb:>9.1f}{r.rtf:>8.3f}{fl_ov:>9.3f}{word_ov:>10.3f}{ngram_ov:>9.3f}")
        summary_rows.append({
            "engine": r.name, "size_mb": r.size_mb,
            "total_s": r.total_s, "audio_s": r.audio_s, "rtf": round(r.rtf, 3),
            "fl_f1_vs_f16": round(fl_ov, 3) if fl_ov == fl_ov else None,
            "word_f1_vs_f16": round(word_ov, 3) if word_ov == word_ov else None,
            "ngram_overlap_vs_f16": round(ngram_ov, 3) if ngram_ov == ngram_ov else None,
            "texts": r.texts, "fls": r.fls,
        })

    out = ROOT / "tests/eval/runs/bench_engines_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"\n[bench] full report → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
