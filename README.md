# CLIP & SigLIP Implementation

这是一个基于 PyTorch 和 Hugging Face `transformers` 库的简单视觉语言模型 (VLM) 训练框架。主要实现了经典 **CLIP** (Contrastive Language-Image Pre-training) 和 **SigLIP** (Sigmoid Loss for Language Image Pre-Training) 架构。

## 特性

- ✅ **CLIP 实现**: 提供标准的双向对比学习损失 (InfoNCE) 进行图文匹配。
- ✅ **SigLIP 实现**: 基于成对 Sigmoid 损失架构，在无需大 Batch Size 全局归一化的前提下达到更优秀的收敛效果。
- ✅ **模块化模型设计**: `model/clip.py` 包含了底层的双塔结构、温度参数和映射层设计。
- ✅ **Hugging Face 支持**: 利用 `transformers` 和 `datasets`，轻松下载在线数据集（如 Flickr30k）以及使用主流大模型的权重初始化（如 `google/vit-base-patch16` 和 `bert-base-uncased`）。
- ✅ **训练脚本分离**: 分别提供 `train.py` 和 `train_siglip.py` 满足两种模型的验证和训练流程。

## 目录结构

```text
├── dataloader.py        # 数据集预处理与加载 (封装了 FlickrDataset)
├── inference.py         # (可选) 模型加载及推理脚本
├── model/
│   ├── __init__.py
│   └── clip.py          # CLIP 与 SigLip 的核心模型定义
├── train.py             # CLIP 模型的训练与验证入口
├── train_siglip.py      # SigLip 模型的训练与验证入口 (定制了优化器与评价指标)
└── README.md            # 项目说明文件
```

## 环境依赖

- Python >= 3.8
- PyTorch >= 2.0
- transformers
- datasets
- matplotlib
- tqdm

安装依赖项示例：
```bash
pip install torch torchvision
pip install transformers datasets matplotlib tqdm
```

## 快速开始

### 1. 数据准备
本项目使用 Hugging Face 的在线数据集（默认通过 `load_dataset` 加载 `lmms-lab/flickr30k`）。
> **注意**：针对国内网络环境，训练脚本中已内置配置 `HF_ENDPOINT = "https://hf-mirror.com"` 加速下载。如果您遇到网络问题，请确保该镜像可用或自备代理。

### 2. 模型训练

本项目提供两套独立的训练流程。

**训练标准 CLIP 模型：**
```bash
python train.py
```
* 输出: 训练曲线 (`loss_curve.png`), 最优模型权重 (`clip_best.pth`)

**训练 SigLIP 模型：**
```bash
python train_siglip.py
```
* 特点: 为温度（`log_t`）和偏置（`log_b`）参数提供了 10 倍增益的独立学习率。
* 输出: 训练曲线 (`loss_curve_siglip.png`), 最优模型权重 (`siglip_best.pth`)

### 3. 模型参数配置
如果需要修改架构（Vision Encoder 或是 Text Encoder），在对应的 `train.py` 或 `train_siglip.py` 中的 `main()` 函数修改以下变量即可：

```python
v_id = "google/vit-base-patch16-224-in21k" # 图像塔
t_id = "bert-base-uncased"                 # 文本塔
h_dim = 512                                # 多模态映射对齐维度
```

## 核心实现亮点

- **Loss 计算差异化**
  - `CLIP`: 使用 `F.cross_entropy` 计算行和列方向的相似度，利用矩阵点乘实现全局图文检索。
  - `SigLIP`: `loss = -torch.nn.functional.logsigmoid(labels * logits).sum() / n`。将多分类问题转化为二分类，用局部成对（Pairwise）的方式替代 Softmax 分母全局的运算，显著降低显存开销，更适合少卡环境。
- **温度参数可学习**
  - 在 `model/clip.py` 中，无论是 `CLIP.logits_scale` 还是 `SigLip.log_t`，温度参数都被封装在 `nn.Parameter` 中动态更新。

## License
MIT License
