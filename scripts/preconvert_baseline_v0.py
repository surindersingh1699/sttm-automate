"""One-shot pre-conversion of `surt-small-turbo-baseline-v0` to all four
on-disk formats the dashboard's engine selector can pick up:

  data/surt-small-turbo-baseline-v0-ct2/             (faster-whisper int8)
  data/surt-small-turbo-baseline-v0.ggml             (whisper.cpp f16)
  data/surt-small-turbo-baseline-v0-q8_0.ggml        (whisper.cpp q8_0)
  data/surt-small-turbo-baseline-v0-mlx/             (mlx-whisper, Apple Silicon)

Run from project root:

  .venv/bin/python scripts/preconvert_baseline_v0.py

Idempotent: skips any format whose output already exists.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_ID = "surindersinghssj/surt-small-turbo-baseline-v0"
SHORT = MODEL_ID.rsplit("/", 1)[-1]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Local working copy of the model with baseline-v0 weights + a patched
# preprocessor_config.json from v3 (baseline-v0's HF repo ships only
# processor_config.json under the newer transformers naming, but the
# converters all hard-require the legacy preprocessor_config.json — we
# patch it in here so each converter can find it without a network round-trip).
# This dir is built externally before this script runs (see README).
LOCAL_MODEL_DIR = DATA_DIR / "_baseline-v0-source"

# All converters accept either a local path or an HF repo id; using the local
# path avoids `hf_hub_download` looking up files that aren't in the upstream
# repo (e.g., preprocessor_config.json).
MODEL_PATH = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_ID

CT2_DIR = DATA_DIR / f"{SHORT}-ct2"
GGML_F16 = DATA_DIR / f"{SHORT}.ggml"
GGML_Q8 = DATA_DIR / f"{SHORT}-q8_0.ggml"
MLX_DIR = DATA_DIR / f"{SHORT}-mlx"


def _step(name: str) -> None:
    print(f"\n{'=' * 64}\n==> {name}\n{'=' * 64}", flush=True)


def _human_bytes(p: Path) -> str:
    if p.is_file():
        n = p.stat().st_size
    elif p.is_dir():
        n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        return "(missing)"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def step_1_ct2() -> None:
    _step(f"1/4  CT2 int8  →  {CT2_DIR.relative_to(PROJECT_ROOT)}")
    if CT2_DIR.exists() and any(CT2_DIR.iterdir()):
        print(f"    ✓ already exists ({_human_bytes(CT2_DIR)}), skipping")
        return
    cmd = [
        str(PROJECT_ROOT / ".venv/bin/ct2-transformers-converter"),
        "--model", MODEL_PATH,
        "--output_dir", str(CT2_DIR),
        "--quantization", "int8",
        # Local-dir mode: all the baseline-v0 weights/config files plus the
        # patched preprocessor_config.json sit in MODEL_PATH, so --copy_files
        # finds them locally (no hf_hub_download call required).
        "--copy_files", "tokenizer.json", "tokenizer_config.json",
                        "preprocessor_config.json", "generation_config.json",
    ]
    print(f"    $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"ct2-transformers-converter failed (exit {rc})")
    print(f"    ✓ done in {time.time() - t0:.1f}s — {_human_bytes(CT2_DIR)}")


def step_2_ggml_f16() -> None:
    _step(f"2/4  GGML f16  →  {GGML_F16.relative_to(PROJECT_ROOT)}")
    if GGML_F16.exists():
        print(f"    ✓ already exists ({_human_bytes(GGML_F16)}), skipping")
        return
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.transcription._whisper_cpp_convert import convert_hf_to_ggml

    # convert_hf_to_ggml calls snapshot_download(repo_id=...) which only accepts
    # HF repo ids (not local paths). The HF cache for baseline-v0 has been
    # pre-patched with vocab.json + tokenizer aux from v3, so the converter
    # finds them after snapshot_download returns.
    print(f"    convert_hf_to_ggml({MODEL_ID!r}, ...)", flush=True)
    t0 = time.time()
    convert_hf_to_ggml(
        MODEL_ID,
        GGML_F16,
        cache_dir=DATA_DIR / "_whisper_cpp_assets",
        use_f16=True,
    )
    print(f"    ✓ done in {time.time() - t0:.1f}s — {_human_bytes(GGML_F16)}")


def step_3_ggml_q8() -> None:
    _step(f"3/4  GGML q8_0  →  {GGML_Q8.relative_to(PROJECT_ROOT)}")
    if GGML_Q8.exists():
        print(f"    ✓ already exists ({_human_bytes(GGML_Q8)}), skipping")
        return
    quantize_bin = shutil.which("whisper-quantize")
    if not quantize_bin:
        raise SystemExit(
            "whisper-quantize binary not on PATH. Install: brew install whisper-cpp"
        )
    if not GGML_F16.exists():
        raise SystemExit(f"need {GGML_F16} for q8 — run step 2 first")
    cmd = [quantize_bin, str(GGML_F16), str(GGML_Q8), "q8_0"]
    print(f"    $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"whisper-quantize failed (exit {rc})")
    print(f"    ✓ done in {time.time() - t0:.1f}s — {_human_bytes(GGML_Q8)}")


def step_4_mlx() -> None:
    _step(f"4/4  MLX 4-bit  →  {MLX_DIR.relative_to(PROJECT_ROOT)}")
    import platform
    if platform.machine() != "arm64":
        print("    ✗ skipping — MLX requires Apple Silicon (arm64)")
        return
    if (MLX_DIR / "weights.safetensors").exists() and (MLX_DIR / "config.json").exists():
        print(f"    ✓ already exists ({_human_bytes(MLX_DIR)}), skipping")
        return
    sys.path.insert(0, str(PROJECT_ROOT))

    import json
    from dataclasses import asdict

    import mlx.core as mx  # type: ignore
    from mlx.utils import tree_flatten  # type: ignore

    from src.transcription import _mlx_convert

    MLX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"    _mlx_convert.convert({MODEL_PATH!r}, dtype=fp16)", flush=True)
    t0 = time.time()
    model = _mlx_convert.convert(MODEL_PATH, mx.float16)
    cfg = asdict(model.dims)
    weights = dict(tree_flatten(model.parameters()))

    # 4-bit quantization, group_size=64 — same defaults as the live engine.
    class _QArgs:
        q_group_size = 64
        q_bits = 4

    print(f"    quantize(q_bits={_QArgs.q_bits}, group_size={_QArgs.q_group_size})", flush=True)
    weights, cfg = _mlx_convert.quantize(weights, cfg, _QArgs)

    mx.save_safetensors(str(MLX_DIR / "weights.safetensors"), weights)
    cfg["model_type"] = "whisper"
    (MLX_DIR / "config.json").write_text(json.dumps(cfg, indent=4))
    print(f"    ✓ done in {time.time() - t0:.1f}s — {_human_bytes(MLX_DIR)}")


def main() -> None:
    print(f"Preparing {MODEL_ID} for sttm-automate")
    print(f"  project root: {PROJECT_ROOT}")
    t_total = time.time()
    step_1_ct2()
    step_2_ggml_f16()
    step_3_ggml_q8()
    step_4_mlx()
    print(f"\n==> ALL DONE in {time.time() - t_total:.1f}s")
    print(f"\nArtifacts:")
    for p in (CT2_DIR, GGML_F16, GGML_Q8, MLX_DIR):
        marker = "✓" if p.exists() else "✗"
        print(f"  {marker}  {p.relative_to(PROJECT_ROOT)}  ({_human_bytes(p)})")


if __name__ == "__main__":
    main()
