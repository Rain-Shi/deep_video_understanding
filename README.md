## Multimodal Video Understanding System

**Authors**: Jinze Shi, Dishen Yang, Tianyao Yu  
**Model**: EVA-CLIP (ViT-g/14) + Llama‑2‑7B‑Chat + LoRA  
**GPU**: Single NVIDIA L4, 24GB VRAM

This project implements a multimodal **video question-answering** system built on top of MiniGPT4-Video. We complete a full **three-stage training pipeline** (image–text alignment → video–text pretraining → video instruction tuning) and train our **own Stage 1/2/3 checkpoints**. The system currently supports short‑video QA via local video upload and natural language queries, and runs stably on a single L4 GPU with 24GB VRAM. A retrieval‑augmented framework for long‑video reasoning and agent‑based orchestration is partially implemented and remains future work.

---

## Key Features

- **Multimodal video QA**: Supports natural language questions over short videos, using both visual frames and optional subtitles.
- **Three‑stage training pipeline**:
  - Stage 1: Image–text alignment (LAION subset)
  - Stage 2: Video–text pretraining (Condensed Movies, CMD)
  - Stage 3: Video instruction tuning (VideoChatGPT)
- **Memory‑efficient training**:
  - Token Pooling (reduces visual sequence length by 75%)
  - LoRA (only tunes qProj / vProj, ~0.5% trainable parameters)
  - Gradient checkpointing (trade compute for memory)
  - `low_resource=True` + 8‑bit quantization
- **Single‑GPU friendly**: All three stages are trained on a single NVIDIA L4 (24GB VRAM). Stage 2 and 3 use batch size = 1 + token pooling, with peak VRAM around 18.5GB.
- **Modular architecture**: Short videos are handled by a direct end‑to‑end pipeline; long videos are designed to use a retrieval‑augmented (RAG) framework with video segmentation, clip summarization and semantic indexing.

---

## 1. Model and Method

### 1.1 Overall Architecture

Our multimodal LLM consists of three main components:

- **Visual Encoder**: EVA‑CLIP (ViT‑g/14) is used to extract high‑dimensional visual features from images and video frames. It is **frozen in all stages**.
- **Language Model**: Llama‑2‑7B‑Chat serves as the language decoder, responsible for understanding questions, fusing multimodal inputs, and generating answers.
- **Projection Layer**: A learnable linear layer that maps EVA‑CLIP features into the Llama‑2 embedding space. This is the main "bridge module" trained in Stage 1 and 2.

On top of this, we apply LoRA to Llama‑2 in Stage 3, inserting low‑rank adapters into the qProj and vProj layers of the attention blocks. This enables efficient adaptation to video instruction‑tuning under tight memory constraints.

### 1.2 Efficiency Strategies

To make training feasible on a single L4 GPU, we adopt several memory‑saving techniques:

- **Token Pooling**: We merge every 4 consecutive visual tokens into one. This reduces the visual sequence length by ~75%, significantly lowering memory and compute.
- **LoRA**: We apply LoRA to Llama‑2 with rank = 64 and alpha = 16, updating only ~0.5% of the parameters.
- **Gradient Checkpointing**: Enabled for both the visual encoder and LLM, allowing us to train longer sequences at the cost of extra computation.
- **Low‑resource settings**:
  - `low_resource: True`
  - `max_context_len` reduced from 3600 (original MiniGPT4‑Video) to 1024 (Stage 2)
  - `max_txt_len` reduced from 256 to 160
  - `batch_size` reduced from 4 to 1

---

## 2. Three‑Stage Training Pipeline

We adopt a curriculum learning strategy where **each stage's checkpoint initializes the next stage**.

### Stage 1: Image–Text Alignment

- **Dataset**: LAION subset  
- **Scale**: ~1,000 image–text pairs  
- **Goal**: Train only the visual‑to‑text projection layer so that Llama‑2 can "see" static visual features.
- **Key hyperparameters**:
  - `batch_size = 4`
  - `max_txt_len = 160`
  - `max_context_len = 512`
- **Output checkpoint**: `stage1_image_align.pth`

### Stage 2: Video–Text Pretraining (Captioning)

- **Dataset**: Condensed Movies (CMD)  
- **Scale**: ~300 video clips  
- **Data format**: video files + descriptive captions  
- **Preprocessing**:
  - `organize_videos.py`: normalize CMD video directory structure (move year‑based subfolders to a flat root and rename files to `.mp4`).
  - `convert_cmd_to_json.py`: merge official `clips.csv` and `descriptions.csv` into a JSON file with fields `image_id` and `caption`.
- **Training strategy**:
  - Sample 4 frames per clip to represent temporal context.
  - Apply Token Pooling over visual tokens.
- **Low‑resource modifications vs. original config**:
  - `max_txt_len: 256 → 160`
  - `max_context_len: 3600 → 1024`
  - `batch_size: 4 → 1`
  - `low_resource: True`
- **Input checkpoint**: `stage1_image_align.pth`  
- **Output checkpoint**: `stage2_video_pretrain.pth`

### Stage 3: Video Instruction Tuning

- **Dataset**: VideoChatGPT instruction set  
- **Original format**: CSV with `video_id`, questions, answers, and metadata  
- **Preprocessing and cleaning**:
  - `convert_csv_to_json2.py`: convert CSV to JSON and normalize fields to `video_id`, `q`, `a`, `length`.
  - `filter_json.py`: keep only entries whose videos actually exist on disk.
  - `clean_stage3_json.py`: handle different key names (`video_id` / `video_name` / `image_id`) and further remove invalid samples.
- **Final scale**: ~3,359 instruction pairs after cleaning.
- **Low‑resource training config**:
  - `batch_size = 1`
  - `max_epoch = 5` (reduced from 50)
  - `iters_per_epoch = 2` (reduced from 1000, for a lightweight run)
  - `length = 20` (controls effective video clip length)
- **Input checkpoint**: `stage2_video_pretrain.pth`  
- **Output checkpoint**: `stage3_video_instruct_final.pth` (used for inference)

---

## 3. Data Overview

### 3.1 Stage 1: LAION Subset

- ~1k image–text pairs sampled from LAION.
- Stored using WebDataset (tar) format to avoid small‑file I/O bottlenecks.

### 3.2 Stage 2: Condensed Movies (CMD)

- ~300 video clips with descriptive captions.
- Original data often references YouTube IDs; we pre‑download videos and filter invalid links.
- During training, 4 frames per clip are sampled and aligned with captions.

### 3.3 Stage 3: VideoChatGPT

- Human‑annotated video‑instruction pairs (Q&A, detailed descriptions, etc.).
- After URL checking, dead‑link filtering, and JSON conversion, we obtain ~3,359 high‑quality instruction samples.
- This cleaning step is critical for avoiding hallucinations caused by missing videos.

---

## 4. System Architecture and Usage

### 4.1 Software Architecture

The system is a modular **multimodal video QA** pipeline:

- **Video Loader**: Loads local video files and performs frame sampling (YouTube URL support is planned but not yet implemented).
- **Subtitle Module (optional)**: Uses Whisper to generate subtitles and align them with the timeline.
- **Visual Encoder**: EVA‑CLIP extracts visual features; Token Pooling compresses the sequence.
- **Multimodal LLM**: Llama‑2‑7B‑Chat + LoRA + projection layer perform multimodal reasoning.
- **Answer Generator**: Produces natural language responses.

Short‑video QA is fully implemented; long‑video QA is designed to follow a "segment → summarize → retrieve → answer" RAG pipeline (see `index.py`, `goldfish_lv.py`), but the full end‑to‑end path is not yet complete.

### 4.2 Environment and Dependencies

- Python 3.9  
- PyTorch 2.x  
- CUDA 11.8  
- Main libraries: `transformers`, `accelerate`, `bitsandbytes`, `decord`, `opencv-python`, `whisper`, `gradio`, etc.  
  (All are specified in `environment.yml`.)

### 4.3 Quick Start (Short‑Video Demo)

1. **Create Conda environment**

```bash
conda env create -f environment.yml
conda activate goldfish
```

2. **Prepare checkpoints**

Place your three checkpoints under `checkpoints/`:

```text
checkpoints/
  stage1_image_align.pth
  stage2_video_pretrain.pth
  stage3_video_instruct_final.pth
```

3. **Run Gradio demo**

```bash
python minigpt4_video_demo.py \
  --ckpt checkpoints/stage3_video_instruct_final.pth \
  --cfg-path test_configs/llama2_test_config.yaml
```

4. **Command‑line inference**

```bash
python minigpt4_video_inference.py \
  --ckpt checkpoints/stage3_video_instruct_final.pth \
  --cfg-path test_configs/llama2_test_config.yaml \
  --video_path path_to_video.mp4 \
  --question "Your question here"
```

---

## 5. Experiments and Observations

All experiments are conducted on Google Cloud Vertex AI with a **single NVIDIA L4 (24GB VRAM)**.

- **Stage 1**: Training loss decreases smoothly from ~4.0 to 1.5, indicating successful alignment between visual and textual spaces.
- **Stage 2 & 3**: Due to `batch_size = 1`, loss curves exhibit noticeable oscillations, but the overall trend is downward, showing that LoRA + Token Pooling still learn effective video representations and instruction‑following behavior.

Ablation on Token Pooling shows:

- Without pooling, long‑video inputs frequently exceed Llama‑2's context window, causing OOM errors or training failures.
- With pooling, VRAM usage remains around 18.5GB, making video LLM training feasible on a single L4.

---

## 6. Limitations and Future Work

- **Long‑video reasoning**: Only the segmentation, clip summarization, and indexing components are implemented; the full retrieval‑augmented QA pipeline is still under development.
- **Agent‑based orchestration**: The coordinator agent and evaluation agent are designed in the system architecture but not yet integrated into the training/inference loop.
- **Data scale**: Due to hardware constraints, all three stages are trained on relatively small subsets (1000 / 300 / 3359 samples). Larger‑scale training is left for future work.

Planned future directions include:

- Completing the end‑to‑end RAG pipeline for long‑video QA.
- Integrating agent‑based coordination and automatic evaluation.
- Evaluating short‑video performance on more benchmarks such as MSVD, MSRVTT, TGIF, and TVQA.
- Optimizing frame sampling and retrieval strategies to improve coverage and answer quality.
