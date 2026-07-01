"""Legacy LoKR helpers retained for save/load interop with LyCORIS/Kohya tools.

The actual LoKR training adapter is provided by the `peft` library
(`peft.tuners.lokr.LoKrModel` / `LoKrConfig`) — see `models/base.py`'s
`configure_adapter`. This module only contains the helpers needed to:

  - convert the trainable PEFT LoKR state dict emitted by the saver into the
    LyCORIS/Kohya `lokr.safetensors` format consumed by ComfyUI LyCORIS loaders
    (`convert_lokr_state_dict_to_lycoris`); and
  - load existing `lokr.safetensors` checkpoints (whether produced by the
    previous in-tree implementation or by PEFT) into a PEFT-wrapped model
    (`load_lokr_state_dict`).
"""

import torch


# Substring tags identifying LoKR adapter parameters (PEFT names lokr_w{1,2},
# lokr_w{1,2}_a, lokr_w{1,2}_b, lokr_t2). Used by the optimizer routing hooks
# in train.py to keep adapter params out of Muon style orthogonalization.
LOKR_PARAM_NAME_KEYWORDS = ('lokr_w', 'lokr_t')


def convert_lokr_state_dict_to_lycoris(state_dict, adapter_config=None, default_prefix='lora_unet_'):
    """Convert a trainable LoKR parameter state dict (PEFT or legacy) to
    LyCORIS/Kohya format.

    Handles key conversion:
      unet.xxx.lokr_w1            -> lora_unet_xxx.lokr_w1
      text_encoder.xxx.lokr_w1    -> lora_te1_xxx.lokr_w1
      text_encoder_2.xxx.lokr_w1  -> lora_te2_xxx.lokr_w1
      model.diffusion_model.xxx   -> lora_unet_xxx

    Keys without a known prefix get `default_prefix` added.

    If `adapter_config` is provided with an `alpha` key, a `.alpha` scalar
    tensor is synthesized for every converted module path so that downstream
    Kohya/LyCORIS loaders can apply the correct scale = alpha / rank. This
    preserves the behavior of the previous in-tree LoKr implementation, which
    registered a per-module `.alpha` buffer.
    """
    alpha_value = None
    if adapter_config is not None and 'alpha' in adapter_config:
        alpha_value = adapter_config['alpha']

    prefix_map = {
        'model.diffusion_model.': 'lora_unet_',
        'diffusion_model.': 'lora_unet_',
        'net.': 'lora_unet_',
        'unet.': 'lora_unet_',
        'text_encoder.': 'lora_te1_',
        'text_encoder_2.': 'lora_te2_',
    }

    new_state_dict = {}
    converted_module_paths = set()
    for key, tensor in state_dict.items():
        suffix_idx = -1
        for keyword in ['.lokr_w', '.lokr_t', '.alpha']:
            idx = key.rfind(keyword)
            if idx > 0:
                suffix_idx = idx
                break

        if suffix_idx < 0:
            continue

        weight_suffix = key[suffix_idx:]
        module_path = key[:suffix_idx]

        converted = False
        for prefix, replacement in prefix_map.items():
            if module_path.startswith(prefix):
                rest = module_path[len(prefix):]
                rest = rest.replace('.', '_')
                new_key = replacement + rest + weight_suffix
                new_state_dict[new_key] = tensor
                module_path_key = new_key if not weight_suffix else new_key[: -len(weight_suffix)]
                converted_module_paths.add(module_path_key)
                converted = True
                break

        if not converted and default_prefix is not None:
            rest = module_path.replace('.', '_')
            new_key = default_prefix + rest + weight_suffix
            new_state_dict[new_key] = tensor
            module_path_key = new_key if not weight_suffix else new_key[: -len(weight_suffix)]
            converted_module_paths.add(module_path_key)
        elif not converted:
            new_state_dict[key] = tensor

    if alpha_value is not None:
        for module_path in converted_module_paths:
            new_state_dict[f'{module_path}.alpha'] = torch.tensor(float(alpha_value))

    return new_state_dict


def _looks_like_lokr_weight(name):
    return any(kw in name for kw in LOKR_PARAM_NAME_KEYWORDS)


def load_lokr_state_dict(model, state_dict):
    """Load a LoKR state dict into a (PEFT-wrapped) model.

    Accepts incoming keys in any of these formats:
      - PEFT native (`<base>.lokr_w1.default`): direct load.
      - diffusion-pipe internal (`<base>.lokr_w1`, i.e. `.default` was stripped
        by `utils/saver.py`): the `.default` suffix is re-appended before matching.
      - LyCORIS/Kohya (`lora_unet_<base_with_underscores>.lokr_w1`): converted
        back to internal format then matched.

    The model is loaded with ``strict=False``: any unmatched keys (e.g. stale
    `.alpha` buffers from the previous in-tree implementation that PEFT does
    not store) are silently dropped.
    """
    model_param_names = set(n for n, p in model.named_parameters() if _looks_like_lokr_weight(n))
    if not model_param_names:
        raise RuntimeError('No LoKr parameters were found in the model. configure_adapter must be called first.')

    # Whether this model is in PEFT LoKrLayout (param names end with `.default`).
    uses_default_suffix = any(n.endswith('.default') for n in model_param_names)

    # Build reverse mapping LyCORIS-style key -> model param name. Both
    # retained (legacy `.lokr_w1`) and PEFT (`.lokr_w1.default`) forms.
    def _lycoris_form(name):
        # drop a trailing `.default` (PEFT) so the comparison uses the same
        # underscore-flattened module path.
        if name.endswith('.default'):
            name = name[:-len('.default')]
        # Weight suffix is the final component; everything before is the path.
        parts = name.rsplit('.', 1)
        if len(parts) != 2:
            return None
        module_path, weight_name = parts
        return module_path.replace('.', '_') + '.' + weight_name

    lycoris_to_model = {}
    for name in model_param_names:
        lkey = _lycoris_form(name)
        if lkey is not None:
            lycoris_to_model[lkey] = name

    prefix_map_strip = [
        ('lora_unet_', ''),
        ('lora_te1_', ''),
        ('lora_te2_', ''),
        ('lora_te_', ''),
        ('model.diffusion_model.', ''),
        ('diffusion_model.', ''),
        ('net.', ''),
        ('unet.', ''),
        ('text_encoder.', ''),
        ('text_encoder_2.', ''),
        ('transformer.', ''),
    ]

    def _candidates(key):
        # Direct key as-is.
        yield key
        # With .default appended (PEFT model expects this).
        if uses_default_suffix and not key.endswith('.default'):
            yield key + '.default'
        # Strip known prefixes, then try direct + lycoris.
        for prefix, replacement in prefix_map_strip:
            if key.startswith(prefix):
                stripped = replacement + key[len(prefix):]
                yield stripped
                if uses_default_suffix and not stripped.endswith('.default'):
                    yield stripped + '.default'
                # LyCORIS underscores -> dots reversal (kohya format).
                if stripped in lycoris_to_model:
                    yield lycoris_to_model[stripped]
                # Try matching via lycoris map by recomputing the lycoris form
                # of the incoming key against our reverse map.
                lkey = _lycoris_form(stripped)
                if lkey is not None and lkey in lycoris_to_model:
                    yield lycoris_to_model[lkey]

    loadable = {}
    for k, v in state_dict.items():
        if not _looks_like_lokr_weight(k) and not k.endswith('.alpha'):
            # Ignore non-LoKR entries (e.g. stray keys from a mixed checkpoint).
            continue
        if k.endswith('.alpha'):
            # PEFT does not store alpha as a buffer; drop these silently.
            continue
        for cand in _candidates(k):
            if cand in model_param_names:
                loadable[cand] = v
                break

    if not loadable:
        raise RuntimeError('No matching LoKr parameters were found in the provided state dict.')
    model.load_state_dict(loadable, strict=False)