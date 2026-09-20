"""Unlabeled question records, deterministic quotas, and immutable run artifacts."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata
import uuid
try:
    from .chat_records import canonical, digest
    from .contract import BOUNDARIES, compile_request, validate_request
except ImportError:
    from chat_records import canonical, digest
    from contract import BOUNDARIES, compile_request, validate_request

SCHEMA = 'openjev-question-v1'
CHECKS = ('grounded', 'task_faithful', 'scope_clear', 'distinct_options',
          'no_answer_hints', 'no_padding', 'language_matches')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_manifest(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(root / name) != expected:
            raise ValueError('artifact_hash_mismatch')
    return manifest


def manifest(root):
    write_json(root / 'manifest.json', {'files': {p.relative_to(root).as_posix(): sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('manifest.json', '.lock')
        and not p.name.endswith('.partial')}})


class Tokenizer:
    """Real tokenizer plus virtual boundary IDs for the shared compiler budget."""
    def __init__(self, path):
        from tokenizers import Tokenizer as RawTokenizer
        path = Path(path)
        path = path / 'tokenizer.json' if path.is_dir() else path
        self.raw = RawTokenizer.from_file(str(path))
        self.fingerprint = sha(path)
        self.boundary_start = self.raw.get_vocab_size(with_added_tokens=True)
        self.unk_token_id = -1
    def encode(self, text, add_special_tokens=False):
        return self.raw.encode(text, add_special_tokens=add_special_tokens).ids
    def convert_tokens_to_ids(self, items):
        return [self.boundary_start + BOUNDARIES.index(x) for x in items]


def allocations(total, weights):
    exact = [total * w / sum(weights) for w in weights]
    result = [math.floor(x) for x in exact]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - result[i]), i))
    for i in order[:total - sum(result)]: result[i] += 1
    return result


def recipes(config):
    count = config['question_count']
    if type(count) is not int or count < 1 or not 0 < config['choice_fraction'] <= 1:
        raise ValueError('invalid_recipe_count')
    rng = random.Random(config['seed'])
    choice_count = allocations(count, [config['choice_fraction'], 1 - config['choice_fraction']])[0]
    buckets = config['choice_buckets']
    expected_min = 2
    for b in buckets:
        if b['min'] != expected_min or not b['min'] <= b['max'] <= 255 or b['weight'] <= 0:
            raise ValueError('invalid_choice_buckets')
        expected_min = b['max'] + 1
    if expected_min != 256: raise ValueError('buckets_must_cover_2_to_255')
    counts = allocations(choice_count, [b['weight'] for b in buckets])
    queues = []
    for b, n in zip(buckets, counts):
        queue = []
        for i in range(n):
            k = 255 if b['max'] == 255 and i < config['exact_255_min'] else rng.randint(b['min'], b['max'])
            queue.append({'type': 'choice', 'k': k, 'bucket': f"{b['min']}-{b['max']}"})
        queues.append(queue)
    queues.append([{'type': 'noul', 'k': 2, 'bucket': 'noul'} for _ in range(count - choice_count)])
    # Interleave buckets so even a short canary attempts the large-K boundary.
    result = []
    while any(queues):
        for queue in queues:
            if queue:
                item = queue.pop(0)
                item.update(language=rng.choice(config['languages']), form=rng.choice(config['forms']),
                            option_style=rng.choice(config['option_styles']), slot=len(result))
                result.append(item)
    return result


def normalized(value):
    text = value if isinstance(value, str) else canonical(value)
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def validate_draft(payload, state, recipe, tokenizer, config):
    if set(payload) != {'questions'} or not isinstance(payload['questions'], dict) or len(payload['questions']) != 1:
        raise ValueError('one_question_required_no_labels')
    request = {'state': deepcopy(state), 'questions': payload['questions']}
    validate_request(request, supported=('choice', 'noul'))
    q = next(iter(payload['questions'].values()))
    if not isinstance(q['instructions'], str) or not q['instructions'].strip() or q['type'] != recipe['type']:
        raise ValueError('invalid_instructions_or_type')
    if q['type'] == 'choice':
        if len(q['criteria']) != recipe['k']: raise ValueError('candidate_count_mismatch')
        names, descriptions = set(), set()
        for name, value in q['criteria'].items():
            n, d = normalized(name), normalized(value)
            if value is None or not n or not d or n in names or d in descriptions:
                raise ValueError('duplicate_or_empty_candidate')
            names.add(n); descriptions.add(d)
            if len(tokenizer.encode(canonical(value))) > config['max_option_tokens']:
                raise ValueError('option_too_long')
    if len(tokenizer.encode(canonical(state))) > config['max_state_tokens']:
        raise ValueError('state_too_long')
    compiled = compile_request(tokenizer, request, max_length=config['max_path_tokens'],
                               max_total_tokens=config['max_total_tokens'])
    lengths = [len(p) for q in compiled for p in q.paths]
    return request, {'max_path_tokens': max(lengths), 'total_path_tokens': sum(lengths),
                     'option_tokens': [len(tokenizer.encode(canonical(v))) for v in q.get('criteria', {}).values()]}


def validate_review(payload, request):
    if set(payload) != {'decision', 'checks', 'issues', 'evidence'}:
        raise ValueError('invalid_review_fields')
    checks = payload['checks']
    if not isinstance(checks, dict) or set(checks) != set(CHECKS) or any(type(v) is not bool for v in checks.values()):
        raise ValueError('invalid_review_checks')
    if payload['decision'] not in ('accept', 'reject'):
        raise ValueError('invalid_review_decision')
    if any(not isinstance(payload[k], list) or any(not isinstance(x, str) or not x.strip() for x in payload[k]) for k in ('issues', 'evidence')):
        raise ValueError('invalid_review_evidence')
    texts = [m['content'] for m in request['state']['messages']]
    if not payload['evidence'] or any(not any(e in text for text in texts) for e in payload['evidence']):
        raise ValueError('review_evidence_not_in_state')
    accepted = payload['decision'] == 'accept'
    if accepted != (all(checks.values()) and not payload['issues']):
        raise ValueError('inconsistent_review')
    return accepted


def make_record(prefix, request, recipe, metrics, audit):
    return {'schema_version': SCHEMA, 'id': 'question_' + digest([prefix['id'], request['questions']]),
            'input': request, 'input_sha256': digest(request),
            'provenance': {k: prefix[k] for k in ('id', 'group_id', 'split', 'source', 'source_family', 'original_prefix_sha256')},
            'recipe': recipe, 'metrics': metrics, 'audit': audit,
            'status': 'llm_reviewed', 'training_eligible': False}


def validate_record(row):
    expected = {'schema_version', 'id', 'input', 'input_sha256', 'provenance', 'recipe', 'metrics',
                'audit', 'status', 'training_eligible'}
    if set(row) != expected or row['schema_version'] != SCHEMA or row['training_eligible'] is not False:
        raise ValueError('invalid_unlabeled_record')
    validate_request(row['input'], supported=('choice', 'noul'))
    p = row['provenance']
    if p['split'] != 'train' or not p['group_id'] or row['input_sha256'] != digest(row['input']):
        raise ValueError('record_identity_or_split_mismatch')
    if row['id'] != 'question_' + digest([p['id'], row['input']['questions']]) or row['status'] != 'llm_reviewed':
        raise ValueError('record_id_or_status_mismatch')
    canonical(row)
    return row
