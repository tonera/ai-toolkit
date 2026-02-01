"""
把 ai-toolkit 训练导出的 Flux2/Flux2-klein LoRA（常见为 ComfyUI 风格前缀 `diffusion_model.`）
转换成 diffusers 的 flux2 LoRA loader 更容易接受的“无前缀”key 形式。

为什么需要：
- ai-toolkit 的 Flux2 保存 LoRA 时会把 key 从 `transformer.*` 改成 `diffusion_model.*`
  （见 `extensions_built_in/diffusion_models/flux2/flux2_model.py:convert_lora_weights_before_save`）
- diffusers 在加载“非 diffusers 格式的 flux2 lora”时会做一次转换，
  转换器通常期望 key 是 `single_blocks.*` / `double_blocks.*` 等（不带 `diffusion_model.` 前缀）

用法：
  python scripts/convert_flux2_lora_aitk_to_diffusers.py \
    --in  /path/to/your_lora.safetensors \
    --out /path/to/your_lora.diffusers_ready.safetensors
"""

import argparse
from collections import OrderedDict

from safetensors.torch import load_file, save_file


def strip_prefixes(state_dict: dict) -> "OrderedDict[str, object]":
    out = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("diffusion_model."):
            new_k = new_k[len("diffusion_model.") :]
        elif new_k.startswith("transformer."):
            # 有些情况下你可能手动做过转换，仍然保守处理一下
            new_k = new_k[len("transformer.") :]
        out[new_k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True, help="input .safetensors lora path")
    p.add_argument("--out", dest="output_path", required=True, help="output .safetensors path")
    args = p.parse_args()

    sd = load_file(args.input_path)
    converted = strip_prefixes(sd)

    meta = OrderedDict()
    meta["aitk.converted_from"] = args.input_path
    meta["aitk.converter"] = "convert_flux2_lora_aitk_to_diffusers.py"
    save_file(converted, args.output_path, metadata=meta)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()

