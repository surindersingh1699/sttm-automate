"""Vendored HF→GGML converter for whisper.cpp.

Refactored from https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-h5-to-ggml.py
(originally a CLI script; turned into `convert_hf_to_ggml()`). Upstream requires a local
clone of openai/whisper for the mel-filter asset; we fetch that single ~4KB file once and
cache it. Keep in sync with upstream if the GGML binary format changes.
"""

from __future__ import annotations

import io
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch


# HF Whisper → whisper.cpp key remap.
_CONV_MAP = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "proj_out.weight": "decoder.proj.weight",
}

_MEL_FILTERS_URL = (
    "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 style BPE byte↔unicode map (needed for the tokenizer block)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _ensure_mel_filters(cache_dir: Path) -> Path:
    """Download openai/whisper's mel_filters.npz once; cache to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "mel_filters.npz"
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"[WhisperCpp] Fetching mel_filters.npz → {out}")
    with urllib.request.urlopen(_MEL_FILTERS_URL, timeout=30) as resp:
        out.write_bytes(resp.read())
    return out


def convert_hf_to_ggml(
    hf_repo_id: str,
    output_path: Path,
    *,
    cache_dir: Path,
    use_f16: bool = True,
) -> Path:
    """Convert a HuggingFace Whisper checkpoint to a whisper.cpp GGML file.

    Writes the binary directly to `output_path` and returns it.
    """
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[WhisperCpp] Snapshotting HF repo '{hf_repo_id}'")
    dir_model = Path(
        snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.safetensors",
                "*.bin",
                "*.model",
            ],
        )
    )

    hparams = json.loads((dir_model / "config.json").read_text(encoding="utf-8"))
    # Some HF Whisper configs ship `max_length` as None or missing.
    if not isinstance(hparams.get("max_length"), int):
        hparams["max_length"] = hparams.get("max_target_positions", 448)

    print("[WhisperCpp] Loading Torch weights")
    model = WhisperForConditionalGeneration.from_pretrained(dir_model)

    n_mels = hparams["num_mel_bins"]
    mel_path = _ensure_mel_filters(cache_dir)
    with np.load(mel_path) as f:
        filters = torch.from_numpy(f[f"mel_{n_mels}"])

    tokens = json.loads((dir_model / "vocab.json").read_text(encoding="utf-8"))

    print(f"[WhisperCpp] Writing GGML → {output_path}")
    with open(output_path, "wb") as fout:
        fout.write(struct.pack("i", 0x67676D6C))  # magic "ggml"
        fout.write(struct.pack("i", hparams["vocab_size"]))
        fout.write(struct.pack("i", hparams["max_source_positions"]))
        fout.write(struct.pack("i", hparams["d_model"]))
        fout.write(struct.pack("i", hparams["encoder_attention_heads"]))
        fout.write(struct.pack("i", hparams["encoder_layers"]))
        fout.write(struct.pack("i", hparams["max_length"]))
        fout.write(struct.pack("i", hparams["d_model"]))
        fout.write(struct.pack("i", hparams["decoder_attention_heads"]))
        fout.write(struct.pack("i", hparams["decoder_layers"]))
        fout.write(struct.pack("i", hparams["num_mel_bins"]))
        fout.write(struct.pack("i", 1 if use_f16 else 0))

        fout.write(struct.pack("i", filters.shape[0]))
        fout.write(struct.pack("i", filters.shape[1]))
        for i in range(filters.shape[0]):
            for j in range(filters.shape[1]):
                fout.write(struct.pack("f", filters[i][j]))

        byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}
        fout.write(struct.pack("i", len(tokens)))
        for key, _ in sorted(tokens.items(), key=lambda x: x[1]):
            text = bytearray([byte_decoder[c] for c in key])
            fout.write(struct.pack("i", len(text)))
            fout.write(text)

        state = model.state_dict()
        for src in state.keys():
            # `proj_out.weight` is tied to the decoder token embedding in HF; skip.
            if src == "proj_out.weight":
                continue

            parts = src.split(".")[1:]
            if len(parts) >= 2 and parts[1] == "layers":
                parts[1] = "blocks"
                sub = ".".join(parts[3:-1])
                if sub == "encoder_attn.k_proj":
                    mapped = "attn.key" if parts[0] == "encoder" else "cross_attn.key"
                else:
                    mapped = _CONV_MAP[sub]
                name = ".".join(parts[:3] + [mapped] + parts[-1:])
            else:
                joined = ".".join(parts)
                name = _CONV_MAP.get(joined, joined)

            data = state[src].squeeze().numpy().astype(np.float16)

            # Conv biases need a trailing singleton dim in GGML.
            if name in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(data.shape[0], 1)

            n_dims = len(data.shape)
            ftype = 1  # 1 = f16
            if use_f16:
                if (
                    n_dims < 2
                    or name in ("encoder.conv1.bias", "encoder.conv2.bias")
                    or name.endswith("positional_embedding")
                ):
                    data = data.astype(np.float32)
                    ftype = 0
            else:
                data = data.astype(np.float32)
                ftype = 0

            name_bytes = name.encode("utf-8")
            fout.write(struct.pack("iii", n_dims, len(name_bytes), ftype))
            for d in range(n_dims):
                fout.write(struct.pack("i", data.shape[n_dims - 1 - d]))
            fout.write(name_bytes)
            data.tofile(fout)

    print(f"[WhisperCpp] Wrote {output_path.stat().st_size / 1e6:.1f} MB")
    return output_path
