# Fine-Tuning Whisper for Gurbani Recognition

This guide covers how to fine-tune OpenAI's Whisper model on Gurbani audio data for improved kirtan recognition accuracy.

## Why Fine-Tune?

Standard Whisper was trained on general Punjabi audio. Gurbani kirtan has unique characteristics:
- Melodic singing style (raag-based) vs normal speech
- Classical Punjabi vocabulary not common in modern speech
- Background instruments (tabla, harmonium, tanpura)
- Specific pronunciation patterns in Gurbani recitation

Fine-tuning on even 10-20 hours of labeled Gurbani audio can dramatically improve recognition.

## Small vs Medium Model

| Aspect | Small (244M) | Medium (769M) |
|--------|-------------|---------------|
| Fine-tune time (T4 GPU) | ~2-4 hours | ~8-12 hours |
| Fine-tune time (A100 GPU) | ~30-60 min | ~2-4 hours |
| Inference RAM | ~1-2 GB | ~3-5 GB |
| 8GB laptop? | Comfortable | Tight but works |
| Base Punjabi accuracy | Good | Better |

**Recommendation:** Start with **small**. Faster iteration, runs well on 8GB laptops. Only move to medium if accuracy is insufficient.

## Step 1: Prepare Training Data

### Data Sources

1. **SikhNet Gurbani Audio** — Thousands of recorded shabads with known text
2. **SGPC Recordings** — Official recordings from Harmandir Sahib
3. **YouTube Kirtan** — Paired with known shabad text from STTM/BaniDB
4. **Nitnem Recordings** — Clean paath recordings with exact text

### Data Format

Each training sample needs:
- **Audio file** (WAV/MP3/FLAC, ideally 16kHz mono)
- **Transcription** (exact Gurmukhi text)

Organize as a CSV or JSON:
```json
{
  "audio": "data/audio/japji_sahib_01.wav",
  "text": "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"
}
```

### Audio Preprocessing

```bash
# Convert to 16kHz mono WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Split long recordings into 10-30 second segments
# (Whisper works best with segments under 30s)
ffmpeg -i long_recording.wav -f segment -segment_time 15 -c copy segment_%03d.wav
```

### Recommended Dataset Size

| Size | Expected Improvement |
|------|---------------------|
| 5 hours | Noticeable improvement for common shabads |
| 10-20 hours | Good accuracy across most kirtan styles |
| 50+ hours | Excellent accuracy, handles diverse raags/singers |

## Step 2: Set Up Training Environment

### Option A: Google Colab (Free GPU)

Best for getting started. Free T4 GPU available.

### Option B: Local GPU

Requires NVIDIA GPU with at least 8GB VRAM for small model.

### Install Dependencies

```bash
pip install --upgrade transformers datasets accelerate evaluate jiwer tensorboard
pip install --upgrade torch torchaudio
```

## Step 3: Fine-Tune with Hugging Face

### Training Script

```python
"""fine_tune_whisper_gurbani.py — Fine-tune Whisper on Gurbani audio."""

import torch
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


# --- Configuration ---
MODEL_SIZE = "openai/whisper-small"  # or "openai/whisper-medium"
OUTPUT_DIR = "./whisper-gurbani-small"
DATA_DIR = "./gurbani_dataset"  # directory with audio + metadata.csv


# --- Load processor and model ---
processor = WhisperProcessor.from_pretrained(MODEL_SIZE, language="pa", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_SIZE)
model.generation_config.language = "pa"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None


# --- Load dataset ---
# Expects a CSV with columns: audio_path, text
# Or use Hugging Face datasets format
dataset = load_dataset(
    "csv",
    data_files={"train": f"{DATA_DIR}/train.csv", "test": f"{DATA_DIR}/test.csv"},
)
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))


# --- Preprocessing ---
def prepare_dataset(batch):
    audio = batch["audio_path"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])


# --- Data collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# --- Metrics ---
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# --- Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,       # reduce to 8 for 8GB VRAM
    gradient_accumulation_steps=1,         # increase to 2 if reducing batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,                        # ~2-4 hours on T4 for small model
    gradient_checkpointing=True,           # saves memory
    fp16=True,                             # mixed precision training
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=225,
    push_to_hub=False,                     # set True to upload to HF Hub
)


# --- Train ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

trainer.train()

# --- Save ---
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
```

## Step 4: Convert to CTranslate2 (for faster-whisper)

After fine-tuning, convert the model so `faster-whisper` can use it:

```bash
pip install ctranslate2

ct2-opus-converter --model whisper-gurbani-small \
    --output_dir whisper-gurbani-small-ct2 \
    --quantization int8
```

## Step 5: Use in STTM Automate

Point the config to your fine-tuned model:

```python
# In src/config.py, change model_size to the path of your converted model:
class WhisperConfig(BaseModel):
    model_size: str = "./whisper-gurbani-small-ct2"  # path to fine-tuned model
```

Or set it at runtime without changing code:

```python
from src.config import config
config.whisper.model_size = "/path/to/whisper-gurbani-small-ct2"
```

## Step 6: Evaluate

Run the existing accuracy test with your fine-tuned model:

```bash
python -m pytest tests/test_accuracy.py -v
```

Or test with real audio:
```python
from faster_whisper import WhisperModel

model = WhisperModel("./whisper-gurbani-small-ct2", device="cpu", compute_type="int8")
segments, info = model.transcribe("test_kirtan.wav", language="pa")
for seg in segments:
    print(f"[{seg.start:.1f}s -> {seg.end:.1f}s] {seg.text}")
```

## Tips for Better Results

1. **Diverse training data** — Include different raagis, raags, tempos, and recording qualities
2. **Clean labels** — Ensure transcriptions exactly match the SGGS text (use BaniDB as ground truth)
3. **Segment carefully** — Each audio clip should be 5-30 seconds with one pangati or verse
4. **Augment data** — Add background noise (tabla, harmonium) to clean paath recordings
5. **Iterative improvement** — Fine-tune, evaluate on real kirtan, collect errors, add more data for those cases
6. **Start with Nitnem banis** — Most commonly recited, easiest to source clean data for

## Cost Estimates

| Platform | GPU | Cost for Small | Cost for Medium |
|----------|-----|----------------|-----------------|
| Google Colab Free | T4 | Free (limited hours) | Free (limited hours) |
| Google Colab Pro | T4/A100 | ~$2-5 | ~$5-15 |
| Lambda Labs | A100 | ~$3-5 | ~$8-15 |
| RunPod | A100 | ~$2-4 | ~$6-12 |
| Local (own GPU) | RTX 3060+ | Free (electricity) | Free (electricity) |
