## Multimodal Video Understanding System

**Authors**: Jinze Shi, Dishen Yang, Tianyao Yu  
**Model**: EVA-CLIP (ViT-g/14) + Llama‑2‑7B‑Chat + LoRA  
**GPU**: Single NVIDIA L4, 24GB VRAM

本项目实现了一个面向**短视频问答**的多模态理解系统，在 MiniGPT4‑Video 官方代码基础上，完整跑通了 **三阶段训练流水线**（图像–文本对齐 → 视频–文本预训练 → 视频指令微调），并训练出我们自己的 Stage1/2/3 checkpoint。系统目前支持上传本地短视频并以自然语言提问，能够在有限显存（单卡 L4 24GB）下稳定完成推理；长视频检索与代理协作框架已完成设计与部分实现，将作为未来工作继续扩展。

---

## 特点概览

- **多模态视频问答**：支持对短视频进行自然语言提问，模型同时利用视觉帧和（可选的）字幕信息进行推理。
- **三阶段训练流水线**：
  - Stage1：图像–文本对齐（LAION 子集）
  - Stage2：视频–文本预训练（Condensed Movies）
  - Stage3：视频指令微调（VideoChatGPT）
- **高效显存优化**：
  - Token Pooling（视觉 token 聚合，序列长度缩短 75%）
  - LoRA（只调 qProj / vProj，约 0.5% 参数可训练）
  - Gradient Checkpointing（换速度换显存）
  - `low_resource=True` + 8‑bit 量化
- **单卡 L4 可训练**：在 24GB VRAM 下完成全部三个阶段训练，Stage2/3 采用 batch size = 1 + token pooling，训练时显存稳定在约 18.5GB。
- **模块化系统架构**：短视频直接端到端推理；长视频设计了检索增强（RAG）框架，当前实现了视频切片、clip 总结和索引构建，完整长视频 QA 流程留作未来工作。

---

## 1. 模型与方法

### 1.1 整体架构

我们的多模态 LLM 由三部分组成：

- **视觉编码器**：EVA‑CLIP (ViT‑g/14)，用来抽取图像/视频帧的高维视觉特征，**在所有阶段均冻结**。
- **语言模型**：Llama‑2‑7B‑Chat，负责理解问题、融合多模态信息并生成自然语言回答。
- **线性映射层（Projection）**：将 EVA‑CLIP 的视觉特征投影到 Llama‑2 的嵌入空间，是 Stage1/2 中主要训练的“桥接模块”。

在此基础上，我们在 Stage3 中对 Llama‑2 施加 LoRA 低秩适配，只微调注意力层中的 qProj 和 vProj，从而在显存可控的前提下适配视频指令任务。

### 1.2 核心效率策略

为了在单卡 L4 上完成训练，我们采用了多种显存优化手段：

- **Token Pooling**：将连续 4 个视觉 token 聚合为 1 个，使视觉序列长度缩短约 75%，显存和时间开销显著下降。
- **LoRA**：对 Llama‑2 进行低秩适配，LoRA rank = 64, alpha = 16，只更新约 0.5% 参数。
- **Gradient Checkpointing**：对视觉和 LLM 部分开启梯度检查点，在 Stage2/3 中允许更长的视频序列。
- **低资源模式**：
  - `low_resource: True`
  - `max_context_len` 从官方 3600 缩到 1024（Stage2）
  - `max_txt_len` 从 256 缩到 160
  - `batch_size` 从 4 降到 1

---

## 2. 三阶段训练流水线

我们采用 curriculum learning 的思路，**每个阶段的输出 checkpoint 作为下一个阶段的初始化**：

### Stage 1：图像–文本对齐（Image–Text Alignment）

- **数据集**：LAION 子集  
- **规模**：约 1,000 条 image–text pair  
- **目标**：只训练视觉→文本的线性映射层，让 Llama‑2 “看见”静态图像特征。  
- **配置要点**：`batch_size = 4`，`max_txt_len = 160`，`max_context_len = 512`。  
- **输出权重**：`stage1_image_align.pth`

### Stage 2：视频–文本预训练（Video Captioning Pretrain）

- **数据集**：Condensed Movies (CMD)  
- **规模**：约 300 个视频 clip  
- **数据形式**：视频文件 + 描述性 caption  
- **预处理**：
  - 利用 `organize_videos.py` 统一视频目录结构（按年份子目录搬运到根目录，并改名为 `.mp4`）
  - 使用 `convert_cmd_to_json.py` 将官方 `clips.csv` + `descriptions.csv` 转成 JSON（`image_id`, `caption`）
- **训练策略**：每个视频采样 4 帧作为时间维度，并使用 Token Pooling 聚合帧内 token。  
- **关键超参（相对官方的低资源改动）**：
  - `max_txt_len: 256 → 160`
  - `max_context_len: 3600 → 1024`
  - `batch_size: 4 → 1`
  - `low_resource: True`
- **输入 checkpoint**：`stage1_image_align.pth`  
- **输出 checkpoint**：`stage2_video_pretrain.pth`

### Stage 3：视频指令微调（Video Instruction Tuning）

- **数据集**：VideoChatGPT 指令数据  
- **原始格式**：CSV (`video_id`, 问题, 答案等字段)  
- **预处理与清洗**：
  - `convert_csv_to_json2.py`：将 CSV 转为 JSON，字段统一为 `video_id`, `q`, `a`, `length`
  - `filter_json.py`：根据本地存在的视频文件过滤 JSON，剔除缺失视频的样本
  - `clean_stage3_json.py`：兼容不同键名（`video_id` / `video_name` / `image_id`），进一步清洗
- **最终规模**：清洗后约 **3,359 条指令样本**  
- **训练配置（低资源）**：
  - `batch_size = 1`
  - `max_epoch = 5`（原 50）
  - `iters_per_epoch = 2`（原 1000，作为小规模实验）
  - `length = 20`（控制视频片段长度）
- **输入 checkpoint**：`stage2_video_pretrain.pth`  
- **输出 checkpoint**：`stage3_video_instruct_final.pth`（用于推理）

---

## 3. 数据说明

### 3.1 Stage1：LAION 子集

- 从 LAION 采样约 1,000 条 image–text pair。  
- 使用 WebDataset（tar）格式存储，减少大量小文件引起的 I/O 瓶颈。

### 3.2 Stage2：Condensed Movies (CMD)

- 从 CMD 数据集中选取约 300 个视频 clip。  
- 由于原始数据多为 YouTube 链接，我们预先下载并过滤失败链接。  
- 训练时对每个 clip 采样 4 帧，并对齐对应 caption。

### 3.3 Stage3：VideoChatGPT

- 原始是 CSV 标注的 video–instruction 对。  
- 经过 URL 校验、死链过滤和 JSON 格式转换后，得到约 3,359 条高质量指令样本。  
- 这一步对减少“视频缺失导致模型幻觉”非常重要。

---

## 4. 系统架构与使用方式

### 4.1 软件架构

系统整体是一个**多模态视频问答系统**：

- Video Loader：负责视频加载与帧采样（目前支持本地短视频，YouTube URL 功能预留但未完成）。  
- Subtitle Module（可选）：调用 Whisper 生成字幕并对齐时间轴。  
- Visual Encoder：EVA‑CLIP 提取视觉特征，Token Pooling 降维。  
- Multimodal LLM：Llama‑2‑7B‑Chat + LoRA + Projection，进行融合推理。  
- Answer Generator：输出自然语言答案。

目前完整实现的是 **短视频 pipeline**；长视频会走“切片 → clip 总结 → 检索 → 回答”的 RAG 路线，这部分架构已有实现的雏形（如 `index.py`, `goldfish_lv.py`），但尚未端到端跑通。

### 4.2 环境与依赖

- Python 3.9  
- PyTorch 2.x  
- CUDA 11.8  
- 主要依赖：`transformers`, `accelerate`, `bitsandbytes`, `decord`, `opencv-python`, `whisper`, `gradio` 等（可通过 `environment.yml` 创建 conda 环境）。

### 4.3 快速开始（短视频 Demo）

1. **创建环境**

```bash
conda env create -f environment.yml
conda activate goldfish
```

2. **准备 checkpoint**

将三个权重放到 `checkpoints/` 目录下：

```text
checkpoints/
  stage1_image_align.pth
  stage2_video_pretrain.pth
  stage3_video_instruct_final.pth
```

3. **运行短视频 Demo**

```bash
python minigpt4_video_demo.py \
  --ckpt checkpoints/stage3_video_instruct_final.pth \
  --cfg-path test_configs/llama2_test_config.yaml
```

4. **命令行推理**

```bash
python minigpt4_video_inference.py \
  --ckpt checkpoints/stage3_video_instruct_final.pth \
  --cfg-path test_configs/llama2_test_config.yaml \
  --video_path path_to_video.mp4 \
  --question "你的问题"
```

---

## 5. 实验与结果概述

在 Google Cloud Vertex AI（单卡 NVIDIA L4, 24GB VRAM）上，我们记录了三阶段训练的损失曲线（见报告 Fig.6–8）：

- **Stage1**：loss 从约 4.0 平稳下降到 1.5，表明线性映射成功对齐视觉特征与文本空间。  
- **Stage2/3**：由于 batch size = 1，loss 曲线抖动明显，但整体趋势仍然下降，说明在 LoRA + Token Pooling 策略下，模型仍能从有限数据中学习有效的视频理解与指令能力。

Token Pooling 的消融实验表明：

- 不使用 Token Pooling 时，长视频场景经常超出 Llama‑2 的上下文窗口，引发显存溢出或训练失败；  
- 启用 Token Pooling 后，训练显存稳定在约 18.5GB，使得在单卡 L4 上训练成为可能。

---

## 6. 限制与未来工作

- **长视频推理未完全实现**：当前只实现了长视频的切片、clip 总结与索引构建，完整的检索增强问答流程仍在开发中。  
- **代理协作与自动评估**：报告中设计了 coordinator agent 和 evaluation agent 的整体架构，但在代码中还未接入完整 pipeline。  
- **数据规模有限**：由于算力限制，三个 stage 都使用了子集规模的数据（1000 / 300 / 3359），未来可在更大数据上重新训练。

未来计划包括：

- 完成长视频 RAG pipeline 的端到端实现；  
- 加入 agent-based 协调与自动评估；  
- 在更多 benchmark（如 MSVD / MSRVTT / TGIF / TVQA）上系统评测短视频性能；  
- 优化帧采样与检索策略，提高信息覆盖率与回答准确性。


