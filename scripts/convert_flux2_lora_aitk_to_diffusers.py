"""
把 ai-toolkit 训练导出的 Flux2/Flux2-klein LoRA 做“确定性的结构清洗”，让 diffusers 更容易加载：
- 保留/补齐 `diffusion_model.` 前缀（让 diffusers 识别为 ai-toolkit LoRA 并触发内部转换）
- 把 `single_transformer_blocks.N.*` -> `single_blocks.N.*`
- 把 `transformer_blocks.N.*` -> `double_blocks.N.*`

这一步的目标是：把 ai-toolkit 的 LoRA key 变成 diffusers 的 flux2 “非 diffusers LoRA 转换器”
更容易识别的输入格式（`single_blocks.*` / `double_blocks.*`）。

为什么需要：
- ai-toolkit 的 Flux2 保存 LoRA 时会把 key 从 `transformer.*` 改成 `diffusion_model.*`
  （见 `extensions_built_in/diffusion_models/flux2/flux2_model.py:convert_lora_weights_before_save`）
- diffusers 在加载 ai-toolkit 的 Flux2 LoRA 时，会检测 `diffusion_model.` 前缀并做一次转换，
  该转换器内部期望 `single_blocks.*` / `double_blocks.*`（随后会自动加回 `transformer.` 前缀并注册 adapter）

用法：
  python scripts/convert_flux2_lora_aitk_to_diffusers.py \
    --in  /home/tonera/project/ai-toolkit/output/sakimi_flux2_klein_9b_style_lora_v2_stable/sakimi_flux2_klein_9b_style_lora_v2_stable.safetensors \
    --out /home/tonera/project/ai-toolkit/output/sakimi_flux2_klein_9b_style_lora_v2_stable/sakimi_flux2_klein_9b_style_lora_v2_stable_diffusers.safetensors
"""

import argparse
from collections import OrderedDict

import re


_RE_SINGLE_TR = re.compile(r"^single_transformer_blocks\.(\d+)\.")
_RE_DOUBLE_TR = re.compile(r"^transformer_blocks\.(\d+)\.")


def _analyze_keys(keys: list[str]) -> dict:
    info = {
        "count": len(keys),
        "has_diffusion_model_prefix": any(k.startswith("diffusion_model.") for k in keys),
        "has_transformer_prefix": any(k.startswith("transformer.") for k in keys),
        "has_single_blocks": any(k.startswith("single_blocks.") for k in keys),
        "has_double_blocks": any(k.startswith("double_blocks.") for k in keys),
        "has_single_transformer_blocks": any(k.startswith("single_transformer_blocks.") for k in keys),
        "has_transformer_blocks": any(k.startswith("transformer_blocks.") for k in keys),
        "has_lora_A": any(".lora_A.weight" in k for k in keys),
        "has_lora_B": any(".lora_B.weight" in k for k in keys),
        "has_lora_down": any(".lora_down.weight" in k for k in keys),
        "has_lora_up": any(".lora_up.weight" in k for k in keys),
        "has_linear1": any(".linear1." in k for k in keys),
        "has_attn_to_q": any(".attn.to_q." in k for k in keys),
    }
    # block index ranges
    def max_idx(prefix: str) -> int | None:
        idxs = []
        for k in keys:
            if k.startswith(prefix):
                try:
                    idxs.append(int(k.split(".")[1]))
                except Exception:
                    pass
        return max(idxs) if idxs else None

    info["max_single_blocks_idx"] = max_idx("single_blocks.")
    info["max_double_blocks_idx"] = max_idx("double_blocks.")
    info["max_single_transformer_blocks_idx"] = max_idx("single_transformer_blocks.")
    info["max_transformer_blocks_idx"] = max_idx("transformer_blocks.")
    return info


def convert_keys(state_dict: dict) -> "OrderedDict[str, object]":
    out = OrderedDict()
    for k, v in state_dict.items():
        new_k = k

        # 1) 统一/补齐前缀：diffusers 用 `diffusion_model.` 来识别 ai-toolkit LoRA 并触发内部转换
        if new_k.startswith("transformer."):
            new_k = "diffusion_model." + new_k[len("transformer.") :]
        elif not new_k.startswith("diffusion_model."):
            # 兼容你之前用旧脚本已经“去前缀”的情况：把前缀补回来
            new_k = "diffusion_model." + new_k

        # 2) blocks 命名对齐到 comfy 风格（diffusers 的 flux2 非diffusers转换器会吃这个）
        prefix = "diffusion_model."
        body = new_k[len(prefix) :] if new_k.startswith(prefix) else new_k

        # single_transformer_blocks.N.xxx -> single_blocks.N.xxx
        m = _RE_SINGLE_TR.match(body)
        if m:
            body = "single_blocks." + body[len(f"single_transformer_blocks.{m.group(1)}.") - 0 :]
            # 上面拼接会重复 N，修正一下
            body = f"single_blocks.{m.group(1)}." + body.split(".", 2)[2]

        # transformer_blocks.N.xxx -> double_blocks.N.xxx
        m = _RE_DOUBLE_TR.match(body)
        if m:
            body = f"double_blocks.{m.group(1)}." + body.split(".", 2)[2]

        new_k = prefix + body if prefix else body

        out[new_k] = v
    return out


def main():
    # 延迟导入：避免在仅做 key 转换/单测导入该模块时强依赖 torch
    from safetensors.torch import load_file, save_file

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True, help="input .safetensors lora path")
    p.add_argument("--out", dest="output_path", required=True, help="output .safetensors path")
    args = p.parse_args()

    sd = load_file(args.input_path)
    keys = list(sd.keys())
    info_before = _analyze_keys(keys)

    converted = convert_keys(sd)
    info_after = _analyze_keys(list(converted.keys()))

    print("[inspect] before:", info_before)
    print("[inspect] after: ", info_after)

    meta = OrderedDict()
    meta["aitk.converted_from"] = args.input_path
    meta["aitk.converter"] = "convert_flux2_lora_aitk_to_diffusers.py"
    save_file(converted, args.output_path, metadata=meta)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()

