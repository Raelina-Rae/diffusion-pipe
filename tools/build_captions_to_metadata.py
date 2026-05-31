#!/usr/bin/env python3
"""
Build or merge captions.json for multi-caption diffusion-pipe training.

Default (no tag dropout): 2 variants per image
  1. full tags
  2. first_n_tags + newline + nl_caption

Optional tag-dropout variants (--dropout-variants N --tag-dropout 0.3):
  Adds N extra captions alternating dropout+nl / nl+dropout (each with a new dropout draw).
  Example: --dropout-variants 4 → 2 base + 4 dropout = 6 total.

Example (same directory as images):
  image_1.jpg
  image_1.txt          # {tags}
  image_1.caption      # {nl_caption}
  image_1.txt8         # optional {first_n_tags}; omit to take first N tags from .txt

Example runs:
  python tools/build_captions_json.py --input D:/data/my_lora
  python tools/build_captions_json.py --input D:/data/my_lora --base-variants tags
  python tools/build_captions_json.py -i D:/data/my_lora --dropout-variants 2 --tag-dropout 0.3
  python tools/build_captions_json.py -i D:/data/my_lora --dropout-variants 4 --tag-dropout 0.3

See --help for all options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

_BASE_VARIANT_TAGS = 'tags'
_BASE_VARIANT_FIRST_N_NL = 'first_n_nl'

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Keep in sync with utils/dataset.py media enumeration skips (sidecars are not media).
_DATASET_SKIP_SUFFIXES = {'.txt', '.npz', '.json', '.parquet', '.bak'}

# Common image extensions (videos: add your extensions or use --include-videos).
_IMAGE_SUFFIXES = {
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif',
}
_VIDEO_SUFFIXES = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.flv',
}


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
    return _join_tags(_split_tags(tags_text, delimiter)[:n], delimiter)


def _tag_dropout(
    tags_text: str,
    rate: float,
    protect_n: int,
    delimiter: str,
    rng: random.Random,
) -> str:
    tags = _split_tags(tags_text, delimiter)
    if not tags:
        return ''
    protected = tags[:protect_n]
    rest = tags[protect_n:]
    if not rest or rate <= 0:
        return _join_tags(tags, delimiter)
    kept = [t for t in rest if rng.random() >= rate]
    return _join_tags(protected + kept, delimiter)


def _per_image_rng(stem: str, base_seed: int | None, variant: int) -> random.Random:
    if base_seed is None:
        seed = int(hashlib.md5(f'{stem}:{variant}'.encode()).hexdigest(), 16) % (2**32)
    else:
        seed = base_seed + (int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100000) * 10 + variant
    return random.Random(seed)


def _read_sidecar(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding='utf-8-sig').strip()


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


def _parse_base_variants(value: str) -> frozenset[str]:
    allowed = {_BASE_VARIANT_TAGS, _BASE_VARIANT_FIRST_N_NL}
    parts = [p.strip().lower() for p in value.split(',') if p.strip()]
    if not parts:
        raise ValueError('base variants list cannot be empty')
    unknown = set(parts) - allowed
    if unknown:
        raise ValueError(f'unknown base variant(s): {sorted(unknown)}; allowed: {sorted(allowed)}')
    return frozenset(parts)


def _build_variants(
    tags: str,
    nl: str | None,
    first_n_tags: str | None,
    tag_dropout_rate: float,
    protect_first_n: int,
    first_n: int,
    delimiter: str,
    stem: str,
    dropout_seed: int | None,
    base_variants: frozenset[str],
    dropout_variants: int,
) -> list[str]:
    variants: list[str] = []

    if _BASE_VARIANT_TAGS in base_variants and tags:
        variants.append(tags)

    first_n_str = first_n_tags
    if first_n_str is None and tags:
        first_n_str = _first_n_from_tags(tags, first_n, delimiter)

    if _BASE_VARIANT_FIRST_N_NL in base_variants and first_n_str and nl:
        variants.append(f'{first_n_str}.\n{nl}')

    if dropout_variants > 0 and tags and nl:
        for i in range(dropout_variants):
            dropped = _tag_dropout(
                tags, tag_dropout_rate, protect_first_n, delimiter,
                _per_image_rng(stem, dropout_seed, i + 1),
            )
            if not dropped:
                continue
            if i % 2 == 0:
                variants.append(f'{dropped}.\n{nl}')
            else:
                variants.append(f'{nl}\n{dropped}')

    return variants


def build_captions_for_directory(
    root: Path,
    sidecar_dir: Path | None,
    tags_ext: str,
    nl_ext: str,
    first_n_tags_ext: str | None,
    first_n: int,
    delimiter: str,
    tag_dropout_rate: float,
    protect_first_n: int,
    dropout_seed: int | None,
    base_variants: frozenset[str],
    dropout_variants: int,
    include_videos: bool,
    require_nl: bool,
    require_tags: bool,
) -> tuple[dict[str, list[str]], list[str]]:
    tags_ext = _norm_ext(tags_ext)
    nl_ext = _norm_ext(nl_ext)
    if first_n_tags_ext:
        first_n_tags_ext = _norm_ext(first_n_tags_ext)

    sidecar_suffixes = {tags_ext, nl_ext}
    if first_n_tags_ext:
        sidecar_suffixes.add(first_n_tags_ext)

    caption_root = sidecar_dir if sidecar_dir is not None else root
    media_files = _collect_media_files(root, include_videos, sidecar_suffixes)
    out: dict[str, list[str]] = {}
    warnings: list[str] = []

    iterator = media_files
    if tqdm is not None:
        iterator = tqdm(media_files, desc='Building captions', unit='file')

    for media_path in iterator:
        stem = media_path.stem
        key = media_path.name

        sidecar_base = caption_root / stem
        tags = _read_sidecar(sidecar_base.with_suffix(tags_ext))
        nl = _read_sidecar(sidecar_base.with_suffix(nl_ext))
        first_n_tags = None
        if first_n_tags_ext:
            first_n_tags = _read_sidecar(sidecar_base.with_suffix(first_n_tags_ext))

        if require_tags and not tags:
            warnings.append(f'{key}: missing tags file *{tags_ext}')
            continue
        if require_nl and not nl:
            warnings.append(f'{key}: missing nl file *{nl_ext}')
            continue
        if not tags and not nl:
            warnings.append(f'{key}: no caption sidecars found')
            continue

        variants = _build_variants(
            tags=tags or '',
            nl=nl,
            first_n_tags=first_n_tags,
            tag_dropout_rate=tag_dropout_rate,
            protect_first_n=protect_first_n,
            first_n=first_n,
            delimiter=delimiter,
            stem=stem,
            dropout_seed=dropout_seed,
            base_variants=base_variants,
            dropout_variants=dropout_variants,
        )

        if not variants:
            warnings.append(f'{key}: no variants produced (check sidecars)')
            continue

        out[key] = variants

    return out, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Build or merge captions.json for multi-caption diffusion-pipe training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', '-i', type=Path, required=True,
        help='Dataset directory containing images/videos (keys use filenames from here).',
    )
    parser.add_argument(
        '--sidecar-dir', type=Path, default=None,
        help='Directory with caption sidecars (default: same as --input). Use a separate folder to avoid extra files in the training directory.',
    )
    parser.add_argument(
        '--output', '-o', type=Path, default=None,
        help='Output captions.json path (default: <input>/captions.json).',
    )
    parser.add_argument(
        '--merge', action='store_true',
        help='Merge into existing captions.json (update/add keys from this run; keep keys not scanned).',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print summary without writing output.',
    )

    parser.add_argument('--tags-ext', default='.txt', help='Extension for full tag captions (default: .txt).')
    parser.add_argument('--nl-ext', default='.caption', help='Extension for natural-language captions (default: .caption).')
    parser.add_argument(
        '--first-n-tags-ext', default=None,
        help='Optional extension for pre-made first-N tags (e.g. .txt8). If omitted, first N tags are taken from tags file.',
    )
    parser.add_argument('--first-n', type=int, default=8, help='Number of leading tags when deriving first_n from tags file (default: 8).')
    parser.add_argument('--delimiter', default=', ', help='Tag delimiter for split/dropout (default: ", ").')

    parser.add_argument(
        '--base-variants', default='tags,first_n_nl',
        help='Comma-separated base captions (default: tags,first_n_nl → 2 variants). '
             'Use "tags" only for a single tag caption, or "first_n_nl" only.',
    )
    parser.add_argument(
        '--dropout-variants', type=int, default=0, metavar='N',
        help='Number of tag-dropout captions to add (default: 0 = disabled). '
             'Each uses a new dropout draw; alternates tags+nl / nl+tags format. '
             'Total variants = base + N (e.g. default base + 4 = 6).',
    )
    parser.add_argument(
        '--tag-dropout', type=float, default=0.0, metavar='RATE',
        help='Per-tag dropout rate on tags after the protected prefix (default: 0 = off). '
             'Required >0 when --dropout-variants > 0 (e.g. 0.3 or 0.1).',
    )
    parser.add_argument(
        '--protect-first-n', type=int, default=8,
        help='Do not apply dropout to the first N tags (default: 8).',
    )
    parser.add_argument(
        '--dropout-seed', type=int, default=None,
        help='Base RNG seed for tag dropout (default: deterministic hash per image).',
    )

    parser.add_argument('--include-videos', action='store_true', help='Include common video extensions.')
    parser.add_argument('--require-tags', action='store_true', default=True, help='Skip images without tags sidecar (default).')
    parser.add_argument('--no-require-tags', action='store_false', dest='require_tags')
    parser.add_argument('--require-nl', action='store_true', help='Skip images without nl sidecar.')
    parser.add_argument('--indent', type=int, default=2, help='JSON indent (default: 2). Use 0 for compact.')

    args = parser.parse_args()

    if not args.input.is_dir():
        print(f'Error: --input is not a directory: {args.input}', file=sys.stderr)
        return 1
    sidecar_dir = args.sidecar_dir
    if sidecar_dir is not None and not sidecar_dir.is_dir():
        print(f'Error: --sidecar-dir is not a directory: {sidecar_dir}', file=sys.stderr)
        return 1

    if args.tag_dropout < 0 or args.tag_dropout > 1:
        print('Error: --tag-dropout must be between 0 and 1', file=sys.stderr)
        return 1
    if args.dropout_variants < 0:
        print('Error: --dropout-variants must be >= 0', file=sys.stderr)
        return 1
    if args.dropout_variants > 0 and args.tag_dropout <= 0:
        print('Error: --tag-dropout must be > 0 when --dropout-variants > 0', file=sys.stderr)
        return 1

    try:
        base_variants = _parse_base_variants(args.base_variants)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1

    output = args.output or (args.input / 'captions.json')
    built, warnings = build_captions_for_directory(
        root=args.input,
        sidecar_dir=sidecar_dir,
        tags_ext=args.tags_ext,
        nl_ext=args.nl_ext,
        first_n_tags_ext=args.first_n_tags_ext,
        first_n=args.first_n,
        delimiter=args.delimiter,
        tag_dropout_rate=args.tag_dropout,
        protect_first_n=args.protect_first_n,
        dropout_seed=args.dropout_seed,
        base_variants=base_variants,
        dropout_variants=args.dropout_variants,
        include_videos=args.include_videos,
        require_nl=args.require_nl,
        require_tags=args.require_tags,
    )

    base_count = len(base_variants)
    expected = base_count + args.dropout_variants
    print(f'Variant layout: {base_count} base ({",".join(sorted(base_variants))}) + '
          f'{args.dropout_variants} dropout = up to {expected} per image (nl required for first_n_nl/dropout)')

    if args.merge and output.is_file():
        with open(output, encoding='utf-8') as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            print(f'Error: existing {output} must be a JSON object', file=sys.stderr)
            return 1
        existing.update(built)
        final = existing
    else:
        final = built

    num_variants = {len(v) for v in final.values()}
    print(f'Images processed: {len(built)}')
    print(f'Output keys: {len(final)}')
    if len(num_variants) == 1:
        print(f'Variants per image: {num_variants.pop()}')
    elif num_variants:
        print(f'Variants per image (mixed): {sorted(num_variants)} — prefer equal counts for diffusion-pipe')

    if warnings:
        print(f'Warnings ({len(warnings)}):')
        for w in warnings[:20]:
            print(f'  {w}')
        if len(warnings) > 20:
            print(f'  ... and {len(warnings) - 20} more')

    if args.dry_run:
        print('Dry run: not writing output.')
        if built:
            sample_key = next(iter(built))
            print(f'Sample key: {sample_key}')
            for i, cap in enumerate(built[sample_key]):
                preview = cap.replace('\n', '\\n')
                if len(preview) > 120:
                    preview = preview[:120] + '...'
                print(f'  [{i}] {preview.encode("ascii", errors="backslashreplace").decode("ascii")}')
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=args.indent or None)
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
