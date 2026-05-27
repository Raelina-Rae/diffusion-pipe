from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
import torch
import torchvision
import imageio
from deepspeed.utils.logging import logger

from utils.common import is_main_process, empty_cuda_cache
from utils.isolate_rng import isolate_rng


@dataclass(frozen=True)
class SamplePrompt:
    prompt: str
    negative_prompt: str = ""


@dataclass(frozen=True)
class SampleConfig:
    width: int
    height: int
    frames: int
    fps: int
    num_inference_steps: int
    guidance_scale: float
    batch_size: int
    seed: int | None
    prompts: list[SamplePrompt]


def _load_sample_cfg(sample_cfg_path: str | Path) -> SampleConfig:
    sample_cfg_path = Path(sample_cfg_path)
    raw = toml.load(sample_cfg_path)

    prompts_raw = raw.get("prompts", [])
    prompts: list[SamplePrompt] = []
    for p in prompts_raw:
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            continue
        prompts.append(SamplePrompt(prompt=prompt, negative_prompt=(p.get("negative_prompt") or "")))

    return SampleConfig(
        width=int(raw.get("width", 1024)),
        height=int(raw.get("height", 1024)),
        frames=int(raw.get("frames", raw.get("num_frames", 1))),
        fps=int(raw.get("fps", 16)),
        num_inference_steps=int(raw.get("num_inference_steps", raw.get("steps", 30))),
        guidance_scale=float(raw.get("guidance_scale", raw.get("cfg", 5.0))),
        batch_size=int(raw.get("batch_size", 1)),
        seed=(int(raw["seed"]) if "seed" in raw and raw["seed"] is not None else None),
        prompts=prompts,
    )


def _make_sample_out_dir(run_dir: str | Path, step: int | None, epoch: int | None) -> Path:
    root = Path(run_dir) / "samples"
    root.mkdir(parents=True, exist_ok=True)

    if step is None and epoch is None:
        name = "sample"
    elif step is None:
        name = f"epoch{epoch}"
    elif epoch is None:
        name = f"step{step}"
    else:
        name = f"step{step}_epoch{epoch}"

    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _gl_style_stem(step: int | None, idx: int) -> str:
    """
    naming includes the global step so you can correlate progress.
    """
    if step is None:
        return f"sample_{idx:04d}"
    return f"{int(step):08d}_{idx:04d}"


def _should_sample(config: dict[str, Any], step: int, epoch: int, finished_epoch: bool) -> bool:
    if not config.get("sample"):
        return False
    if config.get("sample_at_first", False) and step == 1:
        return True

    sample_every_n_steps = config.get("sample_every_n_steps")
    if sample_every_n_steps and sample_every_n_steps > 0:
        if step % int(sample_every_n_steps) == 0:
            return True

    sample_every_n_epochs = config.get("sample_every_n_epochs")
    if finished_epoch and sample_every_n_epochs and sample_every_n_epochs > 0:
        if epoch % int(sample_every_n_epochs) == 0:
            return True

    return False


def maybe_sample_during_training(
    *,
    config: dict[str, Any],
    model: Any,
    model_engine: Any,
    run_dir: str | Path,
    step: int,
    epoch: int,
    finished_epoch: bool,
    disable_block_swap_for_eval: bool,
) -> list[tuple[Path, str]]:
    """
    Called inside the training loop.

    Sampling is best-effort and will log warnings if a model isn't supported.
    """
    if not is_main_process():
        return []
    if not _should_sample(config, step, epoch, finished_epoch):
        return []

    # Pipeline parallel training shards the model; sampling needs the full model on one process.
    if getattr(model_engine, "is_pipe_parallel", False):
        logger.warning("Sampling is skipped because pipeline parallelism is enabled (pipeline_stages > 1).")
        return []

    sample_cfg_path = config.get("sample")
    if not sample_cfg_path:
        return []
    sample_cfg_path = Path(sample_cfg_path)
    if not sample_cfg_path.is_file():
        logger.warning(f"Sample config {sample_cfg_path} does not exist, skipping sampling.")
        return []

    sample_cfg = _load_sample_cfg(sample_cfg_path)
    if len(sample_cfg.prompts) == 0:
        logger.warning("No prompts found in sample config, skipping sampling.")
        return []

    empty_cuda_cache()
    try:
        model.prepare_block_swap_inference(disable_block_swap=disable_block_swap_for_eval)
    except Exception:
        # Not all models implement block swap.
        pass

    model_type = config.get("model", {}).get("type")
    out_dir = _make_sample_out_dir(run_dir, step=step, epoch=epoch if finished_epoch else None)

    saved: list[tuple[Path, str]] = []
    with torch.no_grad(), isolate_rng():
        if model_type == "sdxl":
            saved = _sample_sdxl_in_memory(model, sample_cfg, out_dir, step=step)
        elif model_type in ("cosmos_predict2", "anima"):
            saved = _sample_cosmos_predict2_in_memory(model, sample_cfg, out_dir, step=step)
        else:
            logger.warning(f"Sampling is not implemented for model.type={model_type!r}.")

    try:
        model.prepare_block_swap_training()
    except Exception:
        pass
    empty_cuda_cache()
    return saved


def _sample_sdxl_in_memory(model: Any, sample_cfg: SampleConfig, out_dir: Path, step: int | None) -> list[tuple[Path, str]]:
    """
    Uses the in-memory Diffusers SDXL pipeline held by `models/sdxl.py`.
    """
    pipe = getattr(model, "diffusers_pipeline", None)
    if pipe is None:
        logger.warning("SDXL sampling failed: model.diffusers_pipeline is missing.")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)

    # Temporarily switch to eval for sampling.
    was_training = pipe.unet.training
    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    generator = torch.Generator(device=device)
    if sample_cfg.seed is not None:
        generator.manual_seed(int(sample_cfg.seed))

    img_idx = 0
    saved: list[tuple[Path, str]] = []
    for p in sample_cfg.prompts:
        images = pipe(
            prompt=[p.prompt] * sample_cfg.batch_size,
            negative_prompt=[p.negative_prompt] * sample_cfg.batch_size if p.negative_prompt else None,
            width=sample_cfg.width,
            height=sample_cfg.height,
            num_inference_steps=sample_cfg.num_inference_steps,
            guidance_scale=sample_cfg.guidance_scale,
            generator=generator,
        ).images

        for img in images:
            stem = _gl_style_stem(step, img_idx)
            out_path = out_dir / f"{stem}.png"
            img.save(out_path)
            saved.append((out_path, f"{p.prompt}\nNEG: {p.negative_prompt}".strip()))
            img_idx += 1

    if was_training:
        pipe.unet.train()
        pipe.text_encoder.train()
        pipe.text_encoder_2.train()
    return saved


def _cosmos_text_conditioning(model: Any, prompt: str, negative_prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Import helpers from the model module so we match training behavior.
    from models import cosmos_predict2 as cosmos_mod

    _ensure_cosmos_text_encoder_materialized_for_sampling(model)

    tokenizer = model.tokenizer
    t5_tokenizer = model.t5_tokenizer

    pos_batch = cosmos_mod._tokenize(tokenizer, [prompt])
    neg_batch = cosmos_mod._tokenize(tokenizer, [negative_prompt])
    t5_pos_batch = cosmos_mod._tokenize(t5_tokenizer, [prompt])
    t5_neg_batch = cosmos_mod._tokenize(t5_tokenizer, [negative_prompt])

    pos_emb = cosmos_mod._compute_text_embeddings(model.text_encoder, pos_batch.input_ids, pos_batch.attention_mask, is_generic_llm=model.is_generic_llm)
    neg_emb = cosmos_mod._compute_text_embeddings(model.text_encoder, neg_batch.input_ids, neg_batch.attention_mask, is_generic_llm=model.is_generic_llm)

    # If the model uses the LLM adapter, apply it to both pos/neg embeddings.
    if getattr(model, "use_llm_adapter", False) and getattr(model.transformer, "llm_adapter", None) is not None:
        llm_adapter = model.transformer.llm_adapter
        try:
            llm_adapter_device = next(llm_adapter.parameters()).device
        except StopIteration:
            llm_adapter_device = pos_emb.device
        # Ensure embeddings are on the same device as adapter weights.
        pos_emb = pos_emb.to(llm_adapter_device)
        neg_emb = neg_emb.to(llm_adapter_device)
        pos_emb = llm_adapter(
            source_hidden_states=pos_emb,
            target_input_ids=t5_pos_batch.input_ids.to(llm_adapter_device),
            target_attention_mask=t5_pos_batch.attention_mask.to(llm_adapter_device),
            source_attention_mask=pos_batch.attention_mask.to(llm_adapter_device),
        )
        pos_emb[~t5_pos_batch.attention_mask.bool().to(pos_emb.device)] = 0

        neg_emb = llm_adapter(
            source_hidden_states=neg_emb,
            target_input_ids=t5_neg_batch.input_ids.to(llm_adapter_device),
            target_attention_mask=t5_neg_batch.attention_mask.to(llm_adapter_device),
            source_attention_mask=neg_batch.attention_mask.to(llm_adapter_device),
        )
        neg_emb[~t5_neg_batch.attention_mask.bool().to(neg_emb.device)] = 0

    return pos_emb, neg_emb, pos_batch.attention_mask, neg_batch.attention_mask


def _ensure_cosmos_text_encoder_materialized_for_sampling(model: Any) -> None:
    """
    Some configs initialize the Anima/Cosmos text encoder on meta (especially when using init_empty_weights
    plus lazy loading / offloading). For sampling we need a real module on CPU so tokenization + HF masking works.
    """
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None:
        raise RuntimeError("model.text_encoder is missing")

    try:
        p = next(text_encoder.parameters())
        is_meta = getattr(p, "is_meta", False)
    except StopIteration:
        is_meta = False

    if not is_meta:
        # Make sure it's on CPU to avoid VRAM spikes during sampling.
        try:
            text_encoder.to("cpu")
        except Exception:
            pass
        return

    logger.warning("Text encoder parameters are on meta device; reloading text encoder weights for sampling on CPU.")

    import transformers
    from transformers import T5TokenizerFast, T5EncoderModel, AutoTokenizer, AutoModelForCausalLM
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from utils.common import load_state_dict, iterate_safetensors

    mc = model.model_config
    dtype = mc["dtype"]

    if "t5_path" in mc:
        # Cosmos T5 path
        t5_state_dict = load_state_dict(mc["t5_path"])
        if mc.get("text_encoder_nf4", False):
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quantization_config = None
        te = T5EncoderModel.from_pretrained(
            None,
            config="configs/t5_old/config.json",
            state_dict=t5_state_dict,
            torch_dtype="auto",
            local_files_only=True,
            quantization_config=quantization_config,
        )
        if quantization_config is None and mc.get("text_encoder_fp8", False):
            for name, p in te.named_parameters():
                if p.ndim == 2 and not ("shared" in name or "relative_attention_bias" in name):
                    p.data = p.data.to(torch.float8_e4m3fn)
        model.tokenizer = model.t5_tokenizer  # align with training behavior for Cosmos
        model.text_encoder = te
        model.is_generic_llm = False
    elif "llm_path" in mc:
        llm_path = mc["llm_path"]
        if isinstance(llm_path, str) and Path(llm_path).is_dir():
            # generic Transformers LLM
            model.tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
            text_encoder_full = AutoModelForCausalLM.from_pretrained(llm_path, dtype=dtype, local_files_only=True)
        else:
            # assume Qwen3-0.6b (Anima)
            model.tokenizer = AutoTokenizer.from_pretrained("configs/qwen3_06b", local_files_only=True)
            llm_config = transformers.Qwen3Config.from_pretrained("configs/qwen3_06b", local_files_only=True)
            with init_empty_weights():
                text_encoder_full = transformers.Qwen3ForCausalLM(llm_config)
            for key, tensor in iterate_safetensors(llm_path):
                set_module_tensor_to_device(text_encoder_full, key, device="cpu", dtype=dtype, value=tensor)

        # Training uses the decoder model's `.model` (base) as the encoder here.
        model.text_encoder = text_encoder_full.model
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        model.text_encoder.config.use_cache = False
        model.is_generic_llm = True
    else:
        raise RuntimeError("Missing text encoder path in model config (need t5_path or llm_path)")

    # Ensure it's real and on CPU
    model.text_encoder.to("cpu")
    model.text_encoder.eval()
    model.text_encoder.requires_grad_(False)


def _load_wan_vae_model_for_sampling(model: Any, device: torch.device, dtype: torch.dtype):
    """
    CosmosPredict2Pipeline stores a WanVAE wrapper in `model.vae`.

    In some setups the VAE weights may remain on meta tensors until first use. This helper
    ensures we have a real materialized VAE module on `device` for decoding.
    """
    vae_wrapper = getattr(model, "vae", None)
    if vae_wrapper is None or getattr(vae_wrapper, "model", None) is None:
        raise RuntimeError("model.vae.model is missing")

    vae_model = vae_wrapper.model
    # If any parameter is meta, we need to materialize then load weights.
    try:
        p = next(vae_model.parameters())
        is_meta = getattr(p, "is_meta", False)
    except StopIteration:
        is_meta = False

    if not is_meta:
        return vae_model.to(device), vae_wrapper.scale

    logger.warning("VAE parameters are on meta device; materializing VAE with to_empty() then loading weights.")

    from models.wan.vae2_1 import WanVAE_  # same module used by models/cosmos_predict2.py
    from utils.common import load_state_dict

    vae_path = model.model_config.get("vae_path")
    if not vae_path:
        raise RuntimeError("model.model_config.vae_path is missing (required to load meta VAE weights for sampling)")

    # Recreate config used in models/cosmos_predict2.py:_video_vae
    cfg = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )

    with torch.device("meta"):
        new_vae = WanVAE_(**cfg)

    # Materialize on target device, then load weights.
    new_vae = new_vae.to_empty(device=device)
    sd = load_state_dict(vae_path)
    missing_keys, unexpected_keys = new_vae.load_state_dict(sd, strict=False)
    if len(unexpected_keys) > 0:
        logger.warning(f"Unexpected VAE keys when loading for sampling: {unexpected_keys[:5]} ...")
    if len(missing_keys) > 0:
        logger.warning(f"Missing VAE keys when loading for sampling: {missing_keys[:5]} ...")

    new_vae = new_vae.to(device=device, dtype=dtype).eval()

    # Ensure scale tensors exist on device/dtype
    mean = vae_wrapper.mean.to(device=device, dtype=dtype)
    std = vae_wrapper.std.to(device=device, dtype=dtype)
    scale = [mean, 1.0 / std]
    return new_vae, scale


def _sample_cosmos_predict2_in_memory(model: Any, sample_cfg: SampleConfig, out_dir: Path, step: int | None) -> list[tuple[Path, str]]:
    """
    Rectified-flow Euler sampler for Cosmos Predict2 / Anima.
    Generates either:
      - PNG (frames==1)
      - MP4 (frames>1)
    """
    transformer = getattr(model, "transformer", None)
    vae = getattr(model, "vae", None)
    if transformer is None or vae is None:
        logger.warning("CosmosPredict2/Anima sampling failed: model.transformer or model.vae is missing.")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = model.model_config["dtype"]

    transformer = transformer.to(device)
    transformer.eval()

    # WanVAE wrapper stores `vae.model` and `vae.scale`, but the module may still be meta.
    vae_model, vae_scale = _load_wan_vae_model_for_sampling(model, device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    if sample_cfg.seed is not None:
        generator.manual_seed(int(sample_cfg.seed))

    # Latent sizes: WanVAE spatial compression is 8 (see tools/cosmos_vae_test.py and configs),
    # temporal compression is model-dependent. For simplicity, we assume latent T == frames for frames==1,
    # and otherwise let user provide frames that match the model expectations.
    latent_h = sample_cfg.height // 8
    latent_w = sample_cfg.width // 8
    latent_t = sample_cfg.frames

    dt = 1.0 / float(sample_cfg.num_inference_steps)

    sample_idx = 0
    saved: list[tuple[Path, str]] = []
    for sp in sample_cfg.prompts:
        for _ in range(sample_cfg.batch_size):
            pos_emb, neg_emb, pos_mask, neg_mask = _cosmos_text_conditioning(model, sp.prompt, sp.negative_prompt or "")

            # Move conditioning to device/dtype
            pos_emb = pos_emb.to(device=device, dtype=dtype)
            neg_emb = neg_emb.to(device=device, dtype=dtype)

            # Start from pure noise at t=1
            x = torch.randn((1, transformer.out_channels, latent_t, latent_h, latent_w), device=device, dtype=dtype, generator=generator)

            # padding mask: zeros (no padding) at pixel resolution H/W for prepare_embedded_sequence
            padding_mask = torch.zeros((1, 1, sample_cfg.height, sample_cfg.width), device=device, dtype=dtype)

            for i in range(sample_cfg.num_inference_steps):
                # integrate from t=1 -> 0
                t = 1.0 - (i + 0.5) * dt
                t_tensor = torch.full((1, 1), float(t), device=device, dtype=dtype)

                v_pos = transformer(x, t_tensor, pos_emb, fps=None, padding_mask=padding_mask)
                v_neg = transformer(x, t_tensor, neg_emb, fps=None, padding_mask=padding_mask)

                v = v_neg + sample_cfg.guidance_scale * (v_pos - v_neg)
                x = x - v * dt

            latents = x
            decoded = vae_model.decode(latents, vae_scale).float().clamp_(-1, 1).squeeze(0)  # (C,T,H,W)
            decoded = (decoded + 1) / 2

            if decoded.shape[1] == 1:
                img = decoded[:, 0, ...].clamp(0, 1)
                pil_img = torchvision.transforms.functional.to_pil_image(img.cpu())
                stem = _gl_style_stem(step, sample_idx)
                out_path = out_dir / f"{stem}.png"
                pil_img.save(out_path)
                saved.append((out_path, f"{sp.prompt}\nNEG: {sp.negative_prompt}".strip()))
            else:
                # (C,T,H,W) -> (T,H,W,C) uint8
                video = decoded.permute(1, 2, 3, 0).clamp(0, 1)
                video = (video * 255).to(torch.uint8).cpu().numpy()
                stem = _gl_style_stem(step, sample_idx)
                out_path = out_dir / f"{stem}.mp4"
                imageio.v3.imwrite(out_path, video, fps=sample_cfg.fps)
                saved.append((out_path, f"{sp.prompt}\nNEG: {sp.negative_prompt}".strip()))

            sample_idx += 1
    return saved

