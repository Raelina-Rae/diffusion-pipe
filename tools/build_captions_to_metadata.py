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

_KEEP_TOKENS_SEPARATOR = '|||'


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


def _process_keep_separator(
    tags_text: str, separator: str, delimiter: str,
) -> tuple[str, list[str], list[str]]:
    """Process the keep_tokens separator in tags.

    Returns ``(full_tags, keep_tags, rest_tags)`` where:
    - ``full_tags`` — tags with the separator removed (original if absent)
    - ``keep_tags`` — tags before the separator (empty list if absent)
    - ``rest_tags`` — tags after the separator (all tags if absent)
    """
    if separator not in tags_text:
        return tags_text, [], _split_tags(tags_text, delimiter)
    parts = tags_text.split(separator)
    # Strip stray commas at separator boundaries (e.g. "tag,||| tag2" -> "tag, tag2")
    parts[0] = parts[0].strip().rstrip(',').strip()
    for i in range(1, len(parts)):
        parts[i] = parts[i].strip().lstrip(',').strip()
    keep_tags = _split_tags(parts[0], delimiter)
    rest_tags: list[str] = []
    for part in parts[1:]:
        rest_tags.extend(_split_tags(part, delimiter))
    full_tags = _join_tags(keep_tags + rest_tags, delimiter)
    return full_tags, keep_tags, rest_tags


def _read_sidecar(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding='utf-8-sig').strip() or None


def _load_protected_tags(filepath: Path | None) -> set[str]:
    """Load a set of protected tag strings from a file (one per line).

    Lines starting with ``#`` are treated as comments and blank lines are
    skipped.  Returns an empty set when *filepath* is ``None`` or missing.
    """
    if filepath is None:
        return set()
    if not filepath.is_file():
        print(f'Warning: protected_tags_file not found: {filepath}', file=sys.stderr)
        return set()
    tags: set[str] = set()
    for line in filepath.read_text(encoding='utf-8-sig').splitlines():
        tag = line.strip()
        if tag and not tag.startswith('#'):
            tags.add(tag)
    return tags


def _per_image_rng(stem: str, base_seed: int | None, slot_index: int) -> random.Random:
    """Deterministic RNG scoped to one image + one dropout slot."""
    if base_seed is None:
        seed = int(hashlib.md5(f'{stem}:{slot_index}'.encode()).hexdigest(), 16) % (2 ** 32)
    else:
        offset = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100_000
        seed = base_seed + offset * 100 + slot_index
    return random.Random(seed)


def _apply_dropout(
    tags_text: str,
    rate: float,
    protect_n: int,
    delimiter: str,
    rng: random.Random,
    protected_tags: set[str],
    shuffle: bool,
) -> tuple[list[str], list[str]]:
    """Split tags into (anchor, kept_tail) after optional shuffle + dropout.

    When ``protect_n`` is 0 and the keep_tokens separator (``|||``) is present,
    the separator splits tags into anchor (before) and tail (after).  Otherwise
    the first ``protect_n`` tags are the anchor (separator removed if present)
    and the rest is the tail.  Tail tags are optionally shuffled, then each is
    kept if it is in ``protected_tags`` or with probability ``1 - rate``.
    """
    full_tags, keep_tags, rest_tags = _process_keep_separator(
        tags_text, _KEEP_TOKENS_SEPARATOR, delimiter,
    )

    if protect_n == 0 and keep_tags:
        anchor = keep_tags
        tail = rest_tags
    else:
        all_tags = keep_tags + rest_tags
        anchor = all_tags[:protect_n]
        tail = all_tags[protect_n:]

    if not anchor and not tail:
        return [], []
    if shuffle and tail:
        rng.shuffle(tail)
    if not tail or rate <= 0.0:
        return anchor, tail
    kept = [t for t in tail if t in protected_tags or rng.random() >= rate]
    return anchor, kept


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
    first_n_count: int,
    delimiter: str,
    dropout_rate: float,
    protected_n_tags: int,
    protected_tags: set[str],
    shuffle: bool,
    num_augmentations: int,
    stem: str,
    dropout_seed: int | None,
    caption_version: int,
) -> list[str]:
    variants: list[str] = []

    # Process keep_tokens separator (|||)
    full_tags, keep_tags, rest_tags = _process_keep_separator(
        tags, _KEEP_TOKENS_SEPARATOR, delimiter,
    )

    # --- Base caption 0 ---
    if caption_version >= 2 and full_tags and nl:
        variants.append(f'{full_tags}.\n{nl}')
    elif full_tags:
        variants.append(full_tags)

    # --- Base caption 1: first_n_tags + nl, or just nl ---
    if nl:
        fn = None
        if first_n_tags is not None:
            fn = first_n_tags
        elif first_n_count > 0:
            fn = _first_n_from_tags(full_tags, first_n_count, delimiter)
        elif keep_tags:
            fn = _join_tags(keep_tags, delimiter)
        if fn:
            variants.append(f'{fn}.\n{nl}')
        else:
            variants.append(nl)

    # --- Augmentation captions (dropout) ---
    if num_augmentations <= 0 or not full_tags or not nl:
        return variants

    for aug_idx in range(num_augmentations):
        # tags-first dropout variant
        rng1 = _per_image_rng(stem, dropout_seed, 2 * aug_idx)
        anchor1, kept1 = _apply_dropout(
            tags, dropout_rate, protected_n_tags, delimiter, rng1,
            protected_tags, shuffle,
        )
        dropout_tags1 = _join_tags(anchor1 + kept1, delimiter)
        if dropout_tags1:
            variants.append(f'{dropout_tags1}.\n{nl}')

        # nl-first dropout variant
        rng2 = _per_image_rng(stem, dropout_seed, 2 * aug_idx + 1)
        anchor2, kept2 = _apply_dropout(
            tags, dropout_rate, protected_n_tags, delimiter, rng2,
            protected_tags, shuffle,
        )
        dropout_tags2 = _join_tags(anchor2 + kept2, delimiter)
        if dropout_tags2:
            variants.append(f'{nl}\n{dropout_tags2}')

    return variants


# ---------------------------------------------------------------------------
# Directory processing
# ---------------------------------------------------------------------------

def build_captions_for_directory(
    root: Path,
    sidecar_dir: Path | None,
    tags_ext: str,
    nl_ext: str,
    first_n_tags_ext: str | None,
    first_n_tags_count: int,
    delimiter: str,
    dropout_rate: float,
    protected_n_tags: int,
    protected_tags: set[str],
    shuffle: bool,
    num_augmentations: int,
    dropout_seed: int | None,
    include_videos: bool,
    require_nl: bool,
    require_tags: bool,
    caption_version: int,
) -> tuple[dict[str, list[str]], list[str]]:

    tags_ext  = _norm_ext(tags_ext)
    nl_ext    = _norm_ext(nl_ext)
    if first_n_tags_ext:
        first_n_tags_ext = _norm_ext(first_n_tags_ext)

    # Sidecar extensions should not be treated as media files
    sidecar_suffixes = {tags_ext, nl_ext}
    if first_n_tags_ext:
        sidecar_suffixes.add(first_n_tags_ext)

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
        prebuilt_first_n_tags = _read_sidecar(base.with_suffix(first_n_tags_ext)) if first_n_tags_ext else None

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
            first_n_tags=prebuilt_first_n_tags,
            first_n_count=first_n_tags_count,
            delimiter=delimiter,
            dropout_rate=dropout_rate,
            protected_n_tags=protected_n_tags,
            protected_tags=protected_tags,
            shuffle=shuffle,
            num_augmentations=num_augmentations,
            stem=stem,
            dropout_seed=dropout_seed,
            caption_version=caption_version,
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
    parser.add_argument('--sidecar_dir', type=Path, default=None,
        help='Directory with caption sidecars (default: same as --input).')
    parser.add_argument('--output', '-o', type=Path, default=None,
        help='Output path for captions.json (default: <input>/captions.json).')
    parser.add_argument('--merge', action='store_true',
        help='Merge into existing captions.json instead of overwriting.')
    parser.add_argument('--dry_run', action='store_true',
        help='Print summary and sample output without writing any file.')

    # --- Sidecar extensions ---
    parser.add_argument('--tags_ext', default='.txt',
        help='Extension for full tag captions (default: .txt).')
    parser.add_argument('--nl_ext', default='.caption',
        help='Extension for natural-language captions (default: .caption).')
    parser.add_argument('--first_n_tags_ext', default=None, metavar='EXT',
        help='Extension for pre-built first-N tag file (e.g. .tag). '
             'If omitted, first-N tags are derived from --tags_ext using --first_n_tags.')

    # --- first-N options ---
    parser.add_argument('--first_n_tags', type=int, default=0,
        help='Number of leading tags to use in the first_n_tags base caption '
             'when --first_n_tags_ext is not provided (default: 0 = disabled).')

    # --- Tag delimiter ---
    parser.add_argument('--delimiter', default=', ',
        help='Tag delimiter used for splitting and joining (default: ", ").')

    # --- Caption format ---
    parser.add_argument('--caption_version', type=int, default=2, choices=[1, 2],
        help='Output caption format version (default: 2). '
             'v1: base caption [0] is tags only. '
             'v2: base caption [0] is tags + period + newline + nl.')

    # --- Dropout / augmentation options ---
    parser.add_argument('--num_augmentations', type=int, default=0, metavar='N',
        help='Number of augmentation pairs to generate (default: 0 = disabled). '
             'Each augmentation produces two dropout captions — tags-first and '
             'nl-first — each with an independent dropout mask and optional tag shuffle.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, metavar='RATE',
        help='Per-tag dropout probability applied to tail tags after the protected prefix (default: 0.3). '
             'Ignored when --num_augmentations is 0.')
    parser.add_argument('--protected_n_tags', type=int, default=0,
        help='Number of leading tags that are never dropped or shuffled during dropout (default: 0 = auto-detect keep_tokens separator ||| in tags; if absent, all tags subject to dropout).')
    parser.add_argument('--protected_tags_file', type=Path, default=None, metavar='PATH',
        help='Path to a file listing tags (one per line) that are never dropped during dropout. '
             'Lines starting with # are treated as comments. (default: disabled)')
    parser.add_argument('--shuffle', action='store_true', default=True,
        help='Shuffle non-protected tail tags before each dropout application (default: on). '
             'Only affects augmentation/dropout variants.')
    parser.add_argument('--no_shuffle', action='store_false', dest='shuffle',
        help='Disable tag shuffling for dropout variants.')
    parser.add_argument('--dropout_seed', type=int, default=None,
        help='Base RNG seed for deterministic dropout (default: hash-based per image).')

    # --- Misc ---
    parser.add_argument('--enable_random_caption', action='store_true',
        help='Add __enable_random_caption__ marker to captions.json for the training script.')
    parser.add_argument('--include_videos', action='store_true',
        help='Include common video file extensions alongside images.')
    parser.add_argument('--require_tags', action='store_true', default=True,
        help='Skip images without a tags sidecar (default: on).')
    parser.add_argument('--no_require_tags', action='store_false', dest='require_tags')
    parser.add_argument('--require_nl', action='store_true',
        help='Skip images without a nl sidecar.')
    parser.add_argument('--indent', type=int, default=2,
        help='JSON indentation level (default: 2). Use 0 for compact output.')

    args = parser.parse_args()

    # --- Validation ---
    if not args.input.is_dir():
        print(f'Error: --input is not a directory: {args.input}', file=sys.stderr)
        return 1
    if args.sidecar_dir is not None and not args.sidecar_dir.is_dir():
        print(f'Error: --sidecar_dir is not a directory: {args.sidecar_dir}', file=sys.stderr)
        return 1
    if not (0.0 <= args.dropout_rate <= 1.0):
        print('Error: --dropout_rate must be between 0.0 and 1.0', file=sys.stderr)
        return 1
    if args.num_augmentations < 0:
        print('Error: --num_augmentations must be >= 0', file=sys.stderr)
        return 1
    if args.num_augmentations > 0 and args.dropout_rate <= 0.0:
        print('Error: --dropout_rate must be > 0 when --num_augmentations > 0', file=sys.stderr)
        return 1

    output = args.output or (args.input / 'captions.json')

    # --- Build ---
    protected_tags = _load_protected_tags(args.protected_tags_file)
    if protected_tags:
        print(f'Loaded {len(protected_tags)} protected tags from {args.protected_tags_file}')

    built, warnings = build_captions_for_directory(
        root=args.input,
        sidecar_dir=args.sidecar_dir,
        tags_ext=args.tags_ext,
        nl_ext=args.nl_ext,
        first_n_tags_ext=args.first_n_tags_ext,
        first_n_tags_count=args.first_n_tags,
        delimiter=args.delimiter,
        dropout_rate=args.dropout_rate,
        protected_n_tags=args.protected_n_tags,
        protected_tags=protected_tags,
        shuffle=args.shuffle,
        num_augmentations=args.num_augmentations,
        dropout_seed=args.dropout_seed,
        include_videos=args.include_videos,
        require_nl=args.require_nl,
        require_tags=args.require_tags,
        caption_version=args.caption_version,
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
    base_slots = 2  # tags(+nl), first_n_tags+nl
    total_slots = base_slots + 2 * args.num_augmentations
    print(f'\nCaption layout per image (v{args.caption_version}):')
    if args.caption_version >= 2:
        print(f'  [0] {{tags}}.\\n{{nl}}')
    else:
        print(f'  [0] {{tags}}')
    print(f'  [1] {{first_n_tags}}.\\n{{nl}}')
    for i in range(args.num_augmentations):
        slot_tags = base_slots + 2 * i
        slot_nl   = base_slots + 2 * i + 1
        print(f'  [{slot_tags}] augmentation {i + 1} — dropout tags-first')
        print(f'  [{slot_nl}] augmentation {i + 1} — dropout nl-first')
    parts = [f'dropout rate: {args.dropout_rate}',
             f'protected prefix: {args.protected_n_tags}',
             f'shuffle: {args.shuffle}']
    if protected_tags:
        parts.append(f'protected tags: {len(protected_tags)}')
    print(f'  Total slots: {total_slots}  ({", ".join(parts)})\n')

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
    print(f'\nWrote -> {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())