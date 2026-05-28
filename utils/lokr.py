import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors
from pathlib import Path
from utils.common import get_rank


def factorization(dimension: int, factor: int = -1):
    """Return (m, n) such that m * n == dimension, optimized for balanced factors.
    
    The first value (m) is smaller (for weight scale w1), 
    the second (n) is larger (for weight w2).
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1, w2, scale):
    """Kronecker product of w1 and w2, scaled by scale."""
    if w1.dim() != w2.dim():
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    result = torch.kron(w1, w2)
    if scale != 1:
        result = result * scale
    return result


def rebuild_tucker(t, wa, wb):
    """Rebuild Tucker weight: einsum('ij..., ip, jr -> pr...', t, wa, wb)."""
    return torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)


# LyCORIS preset definitions for module targeting
PRESETS = {
    'attn-mlp': {
        'linear': True,
        'conv2d': False,
    },
    'full': {
        'linear': True,
        'conv2d': True,
    },
}


def get_preset_target_layers(preset_name):
    """Return set of layer type strings for the given preset name."""
    preset = PRESETS.get(preset_name, PRESETS['attn-mlp'])
    targets = set()
    if preset['linear']:
        targets.add('linear')
    if preset['conv2d']:
        targets.add('conv2d')
    return targets


class LoKrModule(nn.Module):
    """LoKr (Low-rank Kronecker Product) module supporting Linear and Conv2d layers.
    
    Replaces ΔW = B·A (LoRA) with ΔW = scale · kron(w1, w2).
    
    Modes:
      - low-rank: w2 = w2_a @ w2_b, optionally w1 = w1_a @ w1_b (decompose_both)
      - full matrix: w1 and/or w2 are full matrices (use_full_matrix)
      - Tucker: conv2d 3x3+ decomposition (use_tucker)
    """
    def __init__(self, org_module, r, alpha, dropout=0.0, factor=-1,
                 full_matrix=False, use_tucker=False, decompose_both=False,
                 org_weight=None, org_bias=None,
                 conv_r=None, conv_alpha=None):
        """LoKr module for Linear and Conv2d layers.

        Args:
            org_module: original nn.Linear or nn.Conv2d
            r: rank for Linear layers
            alpha: scaling alpha for Linear layers
            conv_r: rank for Conv2d layers (falls back to r if None)
            conv_alpha: scaling alpha for Conv2d layers (falls back to alpha if None)
        """
        super().__init__()

        is_conv = isinstance(org_module, nn.Conv2d)
        is_linear = isinstance(org_module, nn.Linear)

        if not is_linear and not is_conv:
            raise TypeError(f'LoKrModule requires nn.Linear or nn.Conv2d, got {type(org_module)}')

        self.module_type = 'conv2d' if is_conv else 'linear'
        self.full_matrix = full_matrix

        if is_conv:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.kernel_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.groups = org_module.groups
            self.tucker = use_tucker and any(k != 1 for k in org_module.kernel_size)
            if conv_r is not None:
                r = conv_r
                alpha = conv_alpha if conv_alpha is not None else conv_r
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.tucker = False

        self.register_buffer('org_weight', (org_weight if org_weight is not None else org_module.weight.data.clone().detach()))
        if is_linear:
            bias = org_bias if org_bias is not None else (org_module.bias.data.clone().detach() if org_module.bias is not None else None)
        else:
            bias = org_bias if org_bias is not None else (org_module.bias.data.clone().detach() if org_module.bias is not None else None)
        if bias is not None:
            self.register_buffer('org_bias', bias)
        else:
            self.org_bias = None

        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        # w1: the "scale" / smaller factor
        if decompose_both and r < max(out_l, in_m) / 2 and not self.full_matrix:
            self.lokr_w1_a = nn.Parameter(torch.empty(out_l, r))
            self.lokr_w1_b = nn.Parameter(torch.empty(r, in_m))
            self.use_w1 = False
        else:
            self.lokr_w1 = nn.Parameter(torch.empty(out_l, in_m))
            self.use_w1 = True

        # w2: the main weight
        if self.module_type == 'conv2d':
            self.shape = (out_dim, in_dim, *self.kernel_size)

            if r >= max(out_k, in_n) / 2 or self.full_matrix:
                if get_rank() == 0 and not self.full_matrix:
                    print(f'LoKr: rank {r} is large for dim={max(in_dim, out_dim)} and factor={factor}, using full matrix mode for Conv2d.')
                self.lokr_w2 = nn.Parameter(torch.empty(out_k, in_n, *self.kernel_size))
                self.use_w2 = True
            elif self.tucker:
                self.lokr_t2 = nn.Parameter(torch.empty(r, r, *self.kernel_size))
                self.lokr_w2_a = nn.Parameter(torch.empty(r, out_k))
                self.lokr_w2_b = nn.Parameter(torch.empty(r, in_n))
                self.use_w2 = False
            else:
                k_prod = 1
                for k in self.kernel_size:
                    k_prod *= k
                self.lokr_w2_a = nn.Parameter(torch.empty(out_k, r))
                self.lokr_w2_b = nn.Parameter(torch.empty(r, in_n * k_prod))
                self.use_w2 = False
        else:
            self.shape = (out_dim, in_dim)
            if r < max(out_k, in_n) / 2 and not self.full_matrix:
                self.lokr_w2_a = nn.Parameter(torch.empty(out_k, r))
                self.lokr_w2_b = nn.Parameter(torch.empty(r, in_n))
                self.use_w2 = False
            else:
                if get_rank() == 0 and not self.full_matrix:
                    print(f'LoKr: rank {r} is large for dim={max(in_dim, out_dim)} and factor={factor}, using full matrix mode for Linear.')
                self.lokr_w2 = nn.Parameter(torch.empty(out_k, in_n))
                self.use_w2 = True

        # Scale: when both w1 and w2 are full matrices, scale = 1
        if self.use_w2 and self.use_w1:
            self.scale = 1.0
        else:
            self.scale = alpha / r

        # Init: make initial ΔW = 0
        if self.use_w1:
            nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            nn.init.constant_(self.lokr_w1_b, 0)

        if self.use_w2:
            nn.init.constant_(self.lokr_w2, 0)
        elif self.tucker:
            nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            nn.init.constant_(self.lokr_w2_b, 0)
        else:
            nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            nn.init.constant_(self.lokr_w2_b, 0)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def get_diff_weight(self):
        w1 = self.lokr_w1 if self.use_w1 else (self.lokr_w1_a @ self.lokr_w1_b)
        if self.use_w2:
            w2 = self.lokr_w2
        elif getattr(self, 'tucker', False) and hasattr(self, 'lokr_t2'):
            w2 = rebuild_tucker(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
        else:
            w2 = self.lokr_w2_a @ self.lokr_w2_b
        result = make_kron(w1, w2, self.scale)
        if self.module_type == 'conv2d' and not self.use_w2 and not getattr(self, 'tucker', False):
            result = result.reshape(*self.shape)
        return result

    def forward(self, x):
        diff = self.get_diff_weight()
        if self.dropout and self.training:
            diff = self.dropout(diff)
        weight = self.org_weight + diff
        if self.module_type == 'linear':
            return F.linear(x, weight, self.org_bias)
        else:
            return F.conv2d(x, weight, self.org_bias,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)


def apply_lokr_to_model(model, adapter_config, target_module_classes=None, preset=None):
    """Apply LoKR to a model by replacing Linear/Conv2d layers within target modules.

    preset:
      - 'attn-mlp' (default): target nn.Linear only
      - 'full': target nn.Linear and nn.Conv2d

    If target_module_classes is provided, only modules with matching class names
    are traversed for layer replacement.

    LoKR-specific config options:
      - preset: 'attn-mlp' (default) or 'full'
      - full_matrix: force full matrix mode (default: False)
      - use_tucker: use Tucker decomposition for Conv2d 3x3+ (default: False)
      - decompose_both: decompose both w1 and w2 (default: False)
      - factor: factorization factor (default: -1)
      - conv_dim: separate rank for Conv2d layers (default: same as rank)
      - conv_alpha: separate alpha for Conv2d layers (default: same as alpha)
    """
    r = adapter_config['rank']
    alpha = adapter_config['alpha']
    dropout = adapter_config.get('dropout', 0.0)
    factor = adapter_config.get('factor', -1)
    full_matrix = adapter_config.get('full_matrix', False)
    use_tucker = adapter_config.get('use_tucker', False)
    decompose_both = adapter_config.get('decompose_both', False)
    conv_r = adapter_config.get('conv_dim', None)
    conv_alpha = adapter_config.get('conv_alpha', None)
    target_layers = get_preset_target_layers(preset or adapter_config.get('preset', 'attn-mlp'))

    for p in model.parameters():
        p.requires_grad_(False)

    replaced = []
    for name, module in model.named_modules():
        if target_module_classes and module.__class__.__name__ not in target_module_classes:
            continue
        for full_name, submodule in module.named_modules(prefix=name):
            is_linear = isinstance(submodule, nn.Linear) and 'linear' in target_layers
            is_conv2d = isinstance(submodule, nn.Conv2d) and 'conv2d' in target_layers
            if not is_linear and not is_conv2d:
                continue

            parts = full_name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            lokr = LoKrModule(
                submodule, r=r, alpha=alpha, dropout=dropout, factor=factor,
                full_matrix=full_matrix, use_tucker=use_tucker,
                decompose_both=decompose_both,
                conv_r=conv_r, conv_alpha=conv_alpha,
            )
            lokr = lokr.to(dtype=submodule.weight.dtype, device=submodule.weight.device)
            setattr(parent, parts[-1], lokr)
            replaced.append((full_name, lokr))

    for name, p in model.named_parameters():
        if any(kw in name for kw in ['lokr_w', 'lokr_t']):
            p.requires_grad_(True)
            p.original_name = name

    return replaced


def get_lokr_state_dict(model):
    """Extract LoKr parameter state dict from a model."""
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoKrModule):
            prefix = name
            if module.use_w1:
                state_dict[f'{prefix}.lokr_w1'] = module.lokr_w1.data
            else:
                state_dict[f'{prefix}.lokr_w1_a'] = module.lokr_w1_a.data
                state_dict[f'{prefix}.lokr_w1_b'] = module.lokr_w1_b.data
            if module.use_w2:
                state_dict[f'{prefix}.lokr_w2'] = module.lokr_w2.data
            elif getattr(module, 'tucker', False) and hasattr(module, 'lokr_t2'):
                state_dict[f'{prefix}.lokr_t2'] = module.lokr_t2.data
                state_dict[f'{prefix}.lokr_w2_a'] = module.lokr_w2_a.data
                state_dict[f'{prefix}.lokr_w2_b'] = module.lokr_w2_b.data
            else:
                state_dict[f'{prefix}.lokr_w2_a'] = module.lokr_w2_a.data
                state_dict[f'{prefix}.lokr_w2_b'] = module.lokr_w2_b.data
    return state_dict


def load_lokr_state_dict(model, state_dict):
    """Load LoKr state dict into a model, handling various prefix formats."""
    model_param_names = set(n for n, p in model.named_parameters() if any(kw in n for kw in ['lokr_w', 'lokr_t']))
    loadable = {}
    for k, v in state_dict.items():
        stripped = re.sub(r'^(transformer|diffusion_model|unet|text_encoder)\.', '', k)
        if stripped in model_param_names:
            loadable[stripped] = v
        elif k in model_param_names:
            loadable[k] = v
    if not loadable:
        raise RuntimeError('No matching LoKr parameters found in state dict')
    model.load_state_dict(loadable, strict=False)
