"""
把 ai-toolkit 训练导出的 Flux2/Flux2-klein LoRA 做“确定性的结构清洗”，让 diffusers 更容易加载：
- 去掉 `diffusion_model.` / `transformer.` 等前缀
- 把 `single_transformer_blocks.N.*` -> `single_blocks.N.*`
- 把 `transformer_blocks.N.*` -> `double_blocks.N.*`

这一步的目标是：把 ai-toolkit 的 LoRA key 变成 diffusers 的 flux2 “非 diffusers LoRA 转换器”
更容易识别的输入格式（`single_blocks.*` / `double_blocks.*`）。

为什么需要：
- ai-toolkit 的 Flux2 保存 LoRA 时会把 key 从 `transformer.*` 改成 `diffusion_model.*`
  （见 `extensions_built_in/diffusion_models/flux2/flux2_model.py:convert_lora_weights_before_save`）
- diffusers 在加载“非 diffusers 格式的 flux2 lora”时会做一次转换，
  转换器通常期望 key 是 `single_blocks.*` / `double_blocks.*` 等（不带 `diffusion_model.` 前缀）

用法：
  python scripts/convert_flux2_lora_aitk_to_diffusers.py \
    --in  /home/tonera/project/ai-toolkit/output/sakimi_flux2_klein_9b_style_lora_v2_stable/sakimi_flux2_klein_9b_style_lora_v2_stable.safetensors \
    --out /home/tonera/project/ai-toolkit/output/sakimi_flux2_klein_9b_style_lora_v2_stable/sakimi_flux2_klein_9b_style_lora_v2_stable_diffusers.safetensors
"""

import argparse
from collections import OrderedDict

import re
from safetensors.torch import load_file, save_file


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

        # 1) 去前缀
        if new_k.startswith("diffusion_model."):
            new_k = new_k[len("diffusion_model.") :]
        if new_k.startswith("transformer."):
            new_k = new_k[len("transformer.") :]

        # 2) blocks 命名对齐到 comfy 风格（diffusers 的 flux2 非diffusers转换器会吃这个）
        # single_transformer_blocks.N.xxx -> single_blocks.N.xxx
        m = _RE_SINGLE_TR.match(new_k)
        if m:
            new_k = "single_blocks." + new_k[len(f"single_transformer_blocks.{m.group(1)}.") - 0 :]
            # 上面拼接会重复 N，修正一下
            new_k = f"single_blocks.{m.group(1)}." + new_k.split(".", 2)[2]

        # transformer_blocks.N.xxx -> double_blocks.N.xxx
        m = _RE_DOUBLE_TR.match(new_k)
        if m:
            new_k = f"double_blocks.{m.group(1)}." + new_k.split(".", 2)[2]

        out[new_k] = v
    return out


def main():
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

