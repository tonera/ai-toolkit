from .flux2_model import Flux2Model
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False

    def load_te(self):
        # 允许从配置覆盖 TE 路径（完全离线/本地镜像场景）
        # 不改变原有逻辑：未提供时仍使用默认的 HF repo（Qwen/Qwen3-*）
        te_path = None
        try:
            # 支持几个常见 key，方便用户配置
            # e.g. model.model_paths.text_encoder: "/models/Qwen3-8B"
            if isinstance(self.model_config.model_paths, dict):
                te_path = (
                    self.model_config.model_paths.get("text_encoder")
                    or self.model_config.model_paths.get("te")
                    or self.model_config.model_paths.get("text_encoder_path")
                )
        except Exception:
            te_path = None

        if te_path is None:
            te_path = self.flux2_klein_te_path

        if te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen3")

        text_encoder: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
            te_path,
            torch_dtype=dtype,
        )
        text_encoder.to(self.device_torch, dtype=dtype)

        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen3")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(te_path)
        return text_encoder, tokenizer


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    # Default to base model filename for backwards-compatibility, but allow distilled too.
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"
    # When `name_or_path` is a local directory or a repo with a different filename, Flux2Model
    # will try these in order and pick the first that exists / downloads successfully.
    flux2_te_filenames: list[str] = [
        "flux-2-klein-base-9b.safetensors",
        "flux-2-klein-9b.safetensors",
    ]

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"
