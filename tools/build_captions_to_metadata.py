from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASET_SKIP_SUFFIXES = {'.txt', '.npz', '.json', '.parquet', '.bak'}

_IMAGE_SUFFIXES = {
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif',
}
_VIDEO_SUFFIXES = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.flv',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        raise ValueError('Extension cannot be empty')
    return ext if ext.startswith('.') else f'.{ext}'


def _split_tags(text: str, delimiter: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in text.split(delimiter) if p.strip()]


def _join_tags(tags: list[str], delimiter: str) -> str:
    return delimiter.join(tags)


def _first_n_from_tags(tags_text: str, n: int, delimiter: str) -> str:
    """Return the first N tags from a tag string."""
    return _join_tags(_split_tags(tags_text, delimiter)[:n], delimiter)


def _read_sidecar(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding='utf-8-sig').strip() or None


def _per_image_rng(stem: str, base_seed: int | None, pair_index: int) -> random.Random:
    """Deterministic RNG scoped to one image + one dropout pair."""
    if base_seed is None:
        seed = int(hashlib.md5(f'{stem}:{pair_index}'.encode()).hexdigest(), 16) % (2 ** 32)
    else:
        offset = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100_000
        seed = base_seed + offset * 100 + pair_index
    return random.Random(seed)


def _apply_dropout(
    tags_text: str,
    rate: float,
    protect_n: int,
    delimiter: str,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Split tags into (protected, kept_remainder) after applying dropout.

    - The first ``protect_n`` tags are always kept (protected).
    - Each remaining tag is kept with probability ``1 - rate``.
    Returns two lists so callers can compose the final caption string freely.
    """
    tags = _split_tags(tags_text, delimiter)
    if not tags:
        return [], []
    protected = tags[:protect_n]
    remainder = tags[protect_n:]
    if not remainder or rate <= 0.0:
        return protected, remainder
    kept = [t for t in remainder if rng.random() >= rate]
    return protected, kept


def _collect_media_files(
    root: Path,
    include_videos: bool,
    extra_skip_suffixes: set[str],
) -> list[Path]:
    skip = _DATASET_SKIP_SUFFIXES | extra_skip_suffixes
    allowed = _IMAGE_SUFFIXES | (_VIDEO_SUFFIXES if include_videos else set())
    files = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in skip:
            continue
        if path.suffix.lower() not in allowed:
            continue
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# Core variant builder
# ---------------------------------------------------------------------------

def _build_variants(
    tags: str,
    nl: str | None,
    first_n_tags: str | None,
    first_n: int,
    delimiter: str,
    dropout_rate: float,
    protect_first_n: int,
    num_dropout: int,
    stem: str,
    dropout_seed: int | None,
) -> list[str]:
    variants: list[str] = []

    # --- Base caption 0: full tags ---
    if tags:
        variants.append(tags)

    # --- Base caption 1: first_n_tags + NL ---
    if nl:
        fn = first_n_tags if first_n_tags is not None else _first_n_from_tags(tags, first_n, delimiter)
        if fn:
            variants.append(f'{fn}.\n{nl}')

    # --- Dropout captions ---
    if num_dropout <= 0 or not tags or not nl:
        return variants

    slots_remaining = num_dropout
    pair_index = 0

    while slots_remaining > 0:
        rng = _per_image_rng(stem, dropout_seed, pair_index)
        protected, kept_remainder = _apply_dropout(tags, dropout_rate, protect_first_n, delimiter, rng)
        all_kept = protected + kept_remainder
        kept_str = _join_tags(all_kept, delimiter)

        # Slot A: tags-first
        if slots_remaining > 0 and kept_str:
            variants.append(f'{kept_str}.\n{nl}')
            slots_remaining -= 1

        # Slot B: nl-first (same mask → same kept_str)
        if slots_remaining > 0 and kept_str:
            variants.append(f'{nl}\n{kept_str}')
            slots_remaining -= 1

        pair_index += 1

    return variants


# ---------------------------------------------------------------------------
# Directory processing
# ---------------------------------------------------------------------------

def build_captions_for_directory(
    root: Path,
    sidecar_dir: Path | None,
    tags_ext: str,
    nl_ext: str,
    first_n_ext: str | None,
    first_n: int,
    delimiter: str,
    dropout_rate: float,
    protect_first_n: int,
    num_dropout: int,
    dropout_seed: int | None,
    include_videos: bool,
    require_nl: bool,
    require_tags: bool,
) -> tuple[dict[str, list[str]], list[str]]:

    tags_ext  = _norm_ext(tags_ext)
    nl_ext    = _norm_ext(nl_ext)
    if first_n_ext:
        first_n_ext = _norm_ext(first_n_ext)

    # Sidecar extensions should not be treated as media files
    sidecar_suffixes = {tags_ext, nl_ext}
    if first_n_ext:
        sidecar_suffixes.add(first_n_ext)

    caption_root = sidecar_dir if sidecar_dir is not None else root
    media_files  = _collect_media_files(root, include_videos, sidecar_suffixes)
    out: dict[str, list[str]] = {}
    warnings: list[str] = []

    iterator = media_files
    if tqdm is not None:
        iterator = tqdm(media_files, desc='Building captions', unit='file')

    for media_path in iterator:
        stem = media_path.stem
        key  = str(media_path)

        base = caption_root / stem
        tags       = _read_sidecar(base.with_suffix(tags_ext))
        nl         = _read_sidecar(base.with_suffix(nl_ext))
        first_n_tags = _read_sidecar(base.with_suffix(first_n_ext)) if first_n_ext else None

        if require_tags and not tags:
            warnings.append(f'{key}: missing tags sidecar (*{tags_ext})')
            continue
        if require_nl and not nl:
            warnings.append(f'{key}: missing nl sidecar (*{nl_ext})')
            continue
        if not tags and not nl:
            warnings.append(f'{key}: no caption sidecars found — skipping')
            continue

        variants = _build_variants(
            tags=tags or '',
            nl=nl,
            first_n_tags=first_n_tags,
            first_n=first_n,
            delimiter=delimiter,
            dropout_rate=dropout_rate,
            protect_first_n=protect_first_n,
            num_dropout=num_dropout,
            stem=stem,
            dropout_seed=dropout_seed,
        )

        if not variants:
            warnings.append(f'{key}: produced no caption variants — skipping')
            continue

        out[key] = variants

    return out, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Build or merge captions.json for multi-caption diffusion-pipe training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Paths ---
    parser.add_argument('--input', '-i', type=Path, required=True,
        help='Dataset directory containing images/videos.')
    parser.add_argument('--sidecar-dir', type=Path, default=None,
        help='Directory with caption sidecars (default: same as --input).')
    parser.add_argument('--output', '-o', type=Path, default=None,
        help='Output path for captions.json (default: <input>/captions.json).')
    parser.add_argument('--merge', action='store_true',
        help='Merge into existing captions.json instead of overwriting.')
    parser.add_argument('--dry-run', action='store_true',
        help='Print summary and sample output without writing any file.')

    # --- Sidecar extensions ---
    parser.add_argument('--tags-ext', default='.txt',
        help='Extension for full tag captions (default: .txt).')
    parser.add_argument('--nl-ext', default='.caption',
        help='Extension for natural-language captions (default: .caption).')
    parser.add_argument('--first-n-ext', default=None, metavar='EXT',
        help='Extension for pre-built first-N tag file (e.g. .txtf). '
             'If omitted, first-N tags are derived from --tags-ext using --first-n.')

    # --- first-N options ---
    parser.add_argument('--first-n', type=int, default=8,
        help='Number of leading tags to use in the first_n_tags base caption '
             'when --first-n-ext is not provided (default: 8).')

    # --- Tag delimiter ---
    parser.add_argument('--delimiter', default=', ',
        help='Tag delimiter used for splitting and joining (default: ", ").')

    # --- Dropout options ---
    parser.add_argument('--num-dropout', type=int, default=0, metavar='N',
        help='Number of additional dropout caption slots to generate (default: 0 = disabled). '
             'Slots are added in pairs: [tags-first, nl-first] sharing the same dropout mask.')
    parser.add_argument('--dropout-rate', type=float, default=0.3, metavar='RATE',
        help='Per-tag dropout probability applied to tags after the protected prefix (default: 0.3). '
             'Ignored when --num-dropout is 0.')
    parser.add_argument('--protect-first-n', type=int, default=8,
        help='Number of leading tags that are never dropped during dropout (default: 8).')
    parser.add_argument('--dropout-seed', type=int, default=None,
        help='Base RNG seed for deterministic dropout (default: hash-based per image).')

    # --- Misc ---
    parser.add_argument('--enable-random-caption', action='store_true',
        help='Add __enable_random_caption__ marker to captions.json for the training script.')
    parser.add_argument('--include-videos', action='store_true',
        help='Include common video file extensions alongside images.')
    parser.add_argument('--require-tags', action='store_true', default=True,
        help='Skip images without a tags sidecar (default: on).')
    parser.add_argument('--no-require-tags', action='store_false', dest='require_tags')
    parser.add_argument('--require-nl', action='store_true',
        help='Skip images without a nl sidecar.')
    parser.add_argument('--indent', type=int, default=2,
        help='JSON indentation level (default: 2). Use 0 for compact output.')

    args = parser.parse_args()

    # --- Validation ---
    if not args.input.is_dir():
        print(f'Error: --input is not a directory: {args.input}', file=sys.stderr)
        return 1
    if args.sidecar_dir is not None and not args.sidecar_dir.is_dir():
        print(f'Error: --sidecar-dir is not a directory: {args.sidecar_dir}', file=sys.stderr)
        return 1
    if not (0.0 <= args.dropout_rate <= 1.0):
        print('Error: --dropout-rate must be between 0.0 and 1.0', file=sys.stderr)
        return 1
    if args.num_dropout < 0:
        print('Error: --num-dropout must be >= 0', file=sys.stderr)
        return 1
    if args.num_dropout > 0 and args.dropout_rate <= 0.0:
        print('Error: --dropout-rate must be > 0 when --num-dropout > 0', file=sys.stderr)
        return 1

    output = args.output or (args.input / 'captions.json')

    # --- Build ---
    built, warnings = build_captions_for_directory(
        root=args.input,
        sidecar_dir=args.sidecar_dir,
        tags_ext=args.tags_ext,
        nl_ext=args.nl_ext,
        first_n_ext=args.first_n_ext,
        first_n=args.first_n,
        delimiter=args.delimiter,
        dropout_rate=args.dropout_rate,
        protect_first_n=args.protect_first_n,
        num_dropout=args.num_dropout,
        dropout_seed=args.dropout_seed,
        include_videos=args.include_videos,
        require_nl=args.require_nl,
        require_tags=args.require_tags,
    )

    # --- Merge ---
    if args.merge and output.is_file():
        with open(output, encoding='utf-8') as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            print(f'Error: existing {output} is not a JSON object', file=sys.stderr)
            return 1
        existing.update(built)
        final = existing
    else:
        final = built

    # --- Summary ---
    base_slots = 2  # tags + first_n_nl
    total_slots = base_slots + args.num_dropout
    print(f'\nCaption layout per image:')
    print(f'  [0] {{tags}}')
    print(f'  [1] {{first_n_tags}}.\\n{{nl_caption}}')
    for i in range(args.num_dropout):
        slot = base_slots + i
        pair = i // 2 + 1
        kind = 'tags-first' if i % 2 == 0 else 'nl-first '
        print(f'  [{slot}] dropout pair {pair} — {kind}')
    print(f'  Total slots: {total_slots}  (dropout rate: {args.dropout_rate}, protect first: {args.protect_first_n})\n')

    data_keys   = [k for k in final if not k.startswith('__')]
    variant_counts = {len(final[k]) for k in data_keys}
    print(f'Images processed : {len(built)}')
    print(f'Output keys      : {len(data_keys)}')
    if len(variant_counts) == 1:
        print(f'Variants/image   : {variant_counts.pop()}')
    elif variant_counts:
        print(f'Variants/image   : mixed {sorted(variant_counts)} — check for missing sidecars')

    if args.enable_random_caption:
        final['__enable_random_caption__'] = True

    if warnings:
        print(f'\nWarnings ({len(warnings)}):')
        for w in warnings[:20]:
            print(f'  {w}')
        if len(warnings) > 20:
            print(f'  … and {len(warnings) - 20} more')

    # --- Dry run preview ---
    if args.dry_run:
        print('\nDry run — nothing written.')
        if built:
            sample_key = next(iter(built))
            print(f'\nSample: {sample_key}')
            for idx, cap in enumerate(built[sample_key]):
                preview = cap.replace('\n', '\\n')
                if len(preview) > 140:
                    preview = preview[:140] + '…'
                print(f'  [{idx}] {preview}')
        return 0

    # --- Write ---
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=args.indent or None)
    print(f'\nWrote → {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())