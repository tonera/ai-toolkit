## Sakimi（风格）— FLUX.2-klein LoRA 训练

### 0) （可选）先把图片文件名统一，方便编辑 captions

如果你现在图片命名很乱，建议先改成 `sakimi_0001.jpg/png...` 这种，后续手工编辑 `.txt` 会省很多事：

```bash
# 先预览（不改名）
python scripts/rename_dataset_files.py \
  --dataset /Users/zhangtao/training/sakimi \
  --prefix sakimi \
  --dry-run

# 确认无误后执行
python scripts/rename_dataset_files.py \
  --dataset /Users/zhangtao/training/sakimi \
  --prefix sakimi \
  --start 1 \
  --pad 4
```

脚本会**同步改名同名的 `.txt`**（如果已经存在），并用“两阶段改名”避免重名覆盖。

### 1) 数据准备（你的数据）

- **图片目录**：`/Users/zhangtao/training/sakimi`
- 建议每张图都配一个同名 `.txt` caption（例如 `001.png` 对应 `001.txt`）
- 你要求的触发词（会被写入每个 caption 的最前面）：
  - `sakimi`
  - `Sakimichan`
  - `Sakimichan style`
  - `Sakimi style`

如果你现在没有 `.txt`，直接运行脚本自动创建/补齐：

```bash
python scripts/prepare_sakimi_dataset.py --dataset /Users/zhangtao/training/sakimi
```

### 2) 训练配置

已生成配置文件：

- `config/train_lora_flux2_klein_sakimi.yaml`

该配置支持用环境变量指定数据集路径（YAML 内写了 `${SAKIMI_DATASET_DIR}` 占位）：

```bash
export SAKIMI_DATASET_DIR=/Users/zhangtao/training/sakimi
```

默认用 **FLUX.2-klein-base-4B**：

- `model.arch: flux2_klein_4b`
- `model.name_or_path: black-forest-labs/FLUX.2-klein-base-4B`

如果你想切到 9B，把这两项改成：

- `model.arch: flux2_klein_9b`
- `model.name_or_path: black-forest-labs/FLUX.2-klein-base-9B`

### 3) 开始训练

直接跑：

```bash
export SAKIMI_DATASET_DIR=/Users/zhangtao/training/sakimi
python run.py config/train_lora_flux2_klein_sakimi.yaml
```

或一键脚本（会先补齐 captions 再开训）：

```bash
bash scripts/train_sakimi_flux2_klein.sh
```

### 4) HF_TOKEN（如模型是 gated）

如果下载模型时报权限/401/403，一般需要 HF token：

- 在项目根目录创建 `.env`
- 写入一行：`HF_TOKEN=你的huggingface_read_token`

### 4.1) （可选）完全离线训练（不从 HF 下载任何东西）

你需要把以下组件都准备到本地：

- **Transformer**：`flux-2-klein-base-9b.safetensors`（在你配置 `model.name_or_path` 指向的目录下）
- **VAE**：`ae.safetensors`（同目录，或在配置里写 `model.vae_path: "/path/to/ae.safetensors"`）
- **Text Encoder（Qwen3）**：本地目录（例如 `/path/to/Qwen3-8B`）

并在 `config/train_lora_flux2_klein_sakimi.yaml` 里加入：

```yaml
model:
  model_paths:
    text_encoder: "/path/to/Qwen3-8B"
```

然后建议设置离线环境变量（让 transformers / HF hub 不走网络）：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

### 5) 推理时触发词怎么用

训练时我们把 4 个触发词都写进 captions 里，所以推理时你可以任选其一或组合，例如：

- `sakimi, 1girl, portrait`
- `Sakimichan style, dramatic lighting`

