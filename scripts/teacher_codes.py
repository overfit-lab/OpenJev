"""Build an existing-token answer codebook; does not load model weights.

Tokenizer paths are runtime inputs and never written into the artifact.
The future collector must validate codes against each actual answer prefix.
"""
from __future__ import annotations
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import string

ANSWER_PREFIX = 'Answer:\n'
AUDIT_PREFIXES = [ANSWER_PREFIX, '</think>\n\n' + ANSWER_PREFIX,
                  '<|im_start|>assistant\n<think>检查候选。\n</think>\n\n' + ANSWER_PREFIX]


def aligned(tokenizer, code, token_id, prefix):
    """Require exact prefix preservation, single-token suffix and round trip."""
    before = tokenizer.encode(prefix, add_special_tokens=False).ids
    after = tokenizer.encode(prefix + code, add_special_tokens=False).ids
    return (after == before + [token_id]
            and tokenizer.decode([token_id], skip_special_tokens=False) == code)


def candidate_pool(tokenizer, prefixes):
    if not prefixes:
        raise ValueError('At least one answer prefix is required')
    pool = []
    for letters in product(string.ascii_uppercase, repeat=2):
        code = ''.join(letters)
        ids = tokenizer.encode(code, add_special_tokens=False).ids
        if len(ids) == 1 and all(aligned(tokenizer, code, ids[0], p) for p in prefixes):
            pool.append({'code': code, 'token_id': ids[0]})
    if len({e['token_id'] for e in pool}) != len(pool):
        raise ValueError('Duplicate token IDs')
    return pool


def select_codes(pool, count, seed):
    if not 2 <= count <= 255 or len(pool) < count:
        raise ValueError('Need 2–255 distinct single-token codes')
    if len({e['code'] for e in pool}) != len(pool) or len({e['token_id'] for e in pool}) != len(pool):
        raise ValueError('Codebook must be one-to-one')
    return random.Random(seed).sample(sorted(pool, key=lambda e: e['code']), count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260920)
    args = parser.parse_args()
    if args.output.is_absolute() or '..' in args.output.parts or args.output.exists():
        parser.error('Use a fresh relative output directory')
    from tokenizers import Tokenizer
    tokenizer_file = args.tokenizer / 'tokenizer.json' if args.tokenizer.is_dir() else args.tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    pool = candidate_pool(tokenizer, AUDIT_PREFIXES)
    selected = select_codes(pool, 255, args.seed)
    numeric_lengths = [len(tokenizer.encode(f'{i:03d}', add_special_tokens=False).ids) for i in range(255)]
    report = {'schema_version': 'openjev-teacher-codebook-v1',
              'tokenizer_sha256': hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
              'answer_prefix': ANSWER_PREFIX, 'tested_prefixes': AUDIT_PREFIXES,
              'tested_two_letter_count': 676, 'valid_two_letter_count': len(pool),
              'selected_count': len(selected), 'seed': args.seed, 'codes': selected,
              'numeric_000_to_254_token_lengths': sorted(set(numeric_lengths)),
              'actual_question_prefix_check_required': True,
              'model_inference_tested': False, 'training_approved': False}
    args.output.mkdir(parents=True)
    (args.output / 'codebook.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest = {'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('tokenizer_sha256', 'valid_two_letter_count', 'selected_count',
                                          'numeric_000_to_254_token_lengths', 'model_inference_tested')}))


if __name__ == '__main__':
    main()
