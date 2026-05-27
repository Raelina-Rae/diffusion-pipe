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
) -> None:
    """
    Called inside the training loop.

    Sampling is best-effort and will log warnings if a model isn't supported.
    """
    if not is_main_process():
        return
    if not _should_sample(config, step, epoch, finished_epoch):
        return

    # Pipeline parallel training shards the model; sampling needs the full model on one process.
    if getattr(model_engine, "is_pipe_parallel", False):
        logger.warning("Sampling is skipped because pipeline parallelism is enabled (pipeline_stages > 1).")
        return

    sample_cfg_path = config.get("sample")
    if not sample_cfg_path:
        return
    sample_cfg_path = Path(sample_cfg_path)
    if not sample_cfg_path.is_file():
        logger.warning(f"Sample config {sample_cfg_path} does not exist, skipping sampling.")
        return

    sample_cfg = _load_sample_cfg(sample_cfg_path)
    if len(sample_cfg.prompts) == 0:
        logger.warning("No prompts found in sample config, skipping sampling.")
        return

    empty_cuda_cache()
    try:
        model.prepare_block_swap_inference(disable_block_swap=disable_block_swap_for_eval)
    except Exception:
        # Not all models implement block swap.
        pass

    model_type = config.get("model", {}).get("type")
    out_dir = _make_sample_out_dir(run_dir, step=step, epoch=epoch if finished_epoch else None)

    with torch.no_grad(), isolate_rng():
        if model_type == "sdxl":
            _sample_sdxl_in_memory(model, sample_cfg, out_dir)
        elif model_type in ("cosmos_predict2", "anima"):
            _sample_cosmos_predict2_in_memory(model, sample_cfg, out_dir)
        else:
            logger.warning(f"Sampling is not implemented for model.type={model_type!r}.")

    try:
        model.prepare_block_swap_training()
    except Exception:
        pass
    empty_cuda_cache()


def _sample_sdxl_in_memory(model: Any, sample_cfg: SampleConfig, out_dir: Path) -> None:
    """
    Uses the in-memory Diffusers SDXL pipeline held by `models/sdxl.py`.
    """
    pipe = getattr(model, "diffusers_pipeline", None)
    if pipe is None:
        logger.warning("SDXL sampling failed: model.diffusers_pipeline is missing.")
        return

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
            img.save(out_dir / f"sample_{img_idx:04d}.png")
            img_idx += 1

    if was_training:
        pipe.unet.train()
        pipe.text_encoder.train()
        pipe.text_encoder_2.train()


def _cosmos_text_conditioning(model: Any, prompt: str, negative_prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Import helpers from the model module so we match training behavior.
    from models import cosmos_predict2 as cosmos_mod

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
        pos_emb = llm_adapter(
            source_hidden_states=pos_emb,
            target_input_ids=t5_pos_batch.input_ids.to(llm_adapter.device),
            target_attention_mask=t5_pos_batch.attention_mask.to(llm_adapter.device),
            source_attention_mask=pos_batch.attention_mask.to(llm_adapter.device),
        )
        pos_emb[~t5_pos_batch.attention_mask.bool().to(pos_emb.device)] = 0

        neg_emb = llm_adapter(
            source_hidden_states=neg_emb,
            target_input_ids=t5_neg_batch.input_ids.to(llm_adapter.device),
            target_attention_mask=t5_neg_batch.attention_mask.to(llm_adapter.device),
            source_attention_mask=neg_batch.attention_mask.to(llm_adapter.device),
        )
        neg_emb[~t5_neg_batch.attention_mask.bool().to(neg_emb.device)] = 0

    return pos_emb, neg_emb, pos_batch.attention_mask, neg_batch.attention_mask


def _sample_cosmos_predict2_in_memory(model: Any, sample_cfg: SampleConfig, out_dir: Path) -> None:
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
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = model.model_config["dtype"]

    transformer = transformer.to(device)
    transformer.eval()

    # WanVAE wrapper stores `vae.model` and `vae.scale`
    vae_model = vae.model.to(device)
    vae_model.eval()

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
            decoded = vae_model.decode(latents, vae.scale).float().clamp_(-1, 1).squeeze(0)  # (C,T,H,W)
            decoded = (decoded + 1) / 2

            if decoded.shape[1] == 1:
                img = decoded[:, 0, ...].clamp(0, 1)
                pil_img = torchvision.transforms.functional.to_pil_image(img.cpu())
                pil_img.save(out_dir / f"sample_{sample_idx:04d}.png")
            else:
                # (C,T,H,W) -> (T,H,W,C) uint8
                video = decoded.permute(1, 2, 3, 0).clamp(0, 1)
                video = (video * 255).to(torch.uint8).cpu().numpy()
                out_path = out_dir / f"sample_{sample_idx:04d}.mp4"
                imageio.v3.imwrite(out_path, video, fps=sample_cfg.fps)

            sample_idx += 1

