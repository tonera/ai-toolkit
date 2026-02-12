# NoobAI（SDXL）角色 LoRA 训练指南（ai-toolkit）

目标：把一组角色图（约 60 张二次元、服装不固定）“融合进 NoobAI（SDXL 底模）”，使推理时**只要提示词出现角色名（触发词）就能稳定画出该角色**，并尽量保持角色一致性且服装可变。

---

## 核心思路

- **训练 SDXL LoRA**（而不是全量微调）：体积小、迁移方便、对底模破坏更小。
- 使用一个**稀有触发词**作为“角色名”，通过 `trigger_word` 注入训练流程。
- 用 `caption_dropout_rate` / `token_dropout_rate` / `keep_tokens` 控制“只写名字也像”与“服装可变”的平衡。
- 可选启用 **DOP（Differential Output Preservation）**减少“概念污染”，让“角色旁边还能出现另一个 1girl”更自然。

---

## 数据集要求（ai-toolkit 约定）

数据集目录包含图片与同名 caption 文本：

- 图片：`.jpg/.jpeg/.png`
- Caption：同名 `.txt`（例如 `001.png` 对应 `001.txt`）
- Caption 文件内容建议使用**逗号分隔的 tag**

ai-toolkit 支持在 caption 中写占位符：

- `"[trigger]"`：会被你的 `trigger_word` 自动替换

并且：如果某张图的 caption 里没有触发词，ai-toolkit 会在训练时自动把触发词加到开头（非 reg 图）。

---

## 触发词（角色名）设计

强烈建议用“稀有 token”，避免与底模已有概念冲突：

- 推荐：`chr_gdBob01`、`zzchr_x9kq01` 这类不常见字符串

训练与推理都使用同一个触发词即可召唤角色。

---

## Caption 写法（固特征绑定 + 服装可变）

你提供的固特征：

- round face
- bob cut
- gold/blonde hair（建议用更常见的 `blonde hair`）
- brown eyes

建议每张图 caption 的“开头骨架”固定：

```text
[trigger], 1girl, round face, bob cut, blonde hair, brown eyes
```

后面再按图补可变项（表情/镜头/姿势/场景/服装）。

### 服装不固定怎么做

- 服装 tag 用更泛化的词（`hoodie/dress/armor/jacket`），少写非常细的款式细节。
- 避免某一类服装在 60 张里占比过高（否则触发词会强绑定那类衣服）。

---

## 关键训练参数（为什么这样配）

### 让“只写名字也像”

- `caption_dropout_rate > 0`：让一部分样本 caption 变为空字符串；此时训练时往往只剩下自动注入的触发词（或 default caption），从而让模型学会“只靠名字召唤”。

### 让“服装可变”

- `token_dropout_rate > 0`：随机丢弃 caption 里的 token，削弱模型对可变 token（尤其服装）的依赖。
- `keep_tokens = N`：保证前 N 个 token 永不被 token dropout 丢掉（通常放触发词 + 角色固特征），让“角色本体”更稳。

建议：

- `keep_tokens = 6`（`[trigger] + 1girl + 4 个固特征`）
- `token_dropout_rate = 0.15` 起步（服装仍粘连就升到 `0.2~0.3`）
- `caption_dropout_rate = 0.05` 起步（只写名字不稳就升到 `0.08~0.12`）

---

## DOP（差分输出保持）注意事项

启用 DOP（`train.diff_output_preservation: true`）时：

- 需要设置 `train.diff_output_preservation_class`，对于二次元角色通常用 `1girl`。
- **不能同时启用** `train.blank_prompt_preservation`（二者互斥）。
- **不能启用** `train.cache_text_embeddings`（ai-toolkit 会报错）。

---

## 训练与验证建议

### 训练后快速验收 3 组 prompt

1. **只写名字**
   - `chr_gdBob01`
2. **名字 + 换装**
   - `chr_gdBob01, 1girl, armor`
   - `chr_gdBob01, 1girl, hoodie`
3. **名字 + 换构图**
   - `chr_gdBob01, 1girl, full body, dynamic pose`

### 常见问题排查

- **只写名字不够像**：提高 `caption_dropout_rate`；或增大 `network.linear`；仍不稳再小心开启 `train_text_encoder`（小学习率、少步数）。
- **衣服粘住/背景粘住**：提高 `token_dropout_rate`，同时清洗数据分布（某类服装/背景不要占比过高）。
- **过拟合（像复刻训练图）**：减少 `steps` 或降低 `lr`。

---

## 参考配置

示例配置见：`config/train_lora_noobai_character_example.yaml`

