"""Current conversation prefixes, privacy filtering and teacher response parsing."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import re

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def sanitize_state(state):
    """Stable per-value placeholders keep different contacts distinct; no mapping is exported."""
    patterns = [
        ('TOKEN', r'\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{12,})\b'),
        ('CREDENTIAL', r'(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token)\s*[:=]\s*[\x22\x27]?[^\s,;\x22\x27]+'),
        ('EMAIL', r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),
        ('IP', r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])'),
        ('PHONE', r'(?<!\d)1[3-9]\d{9}(?!\d)'),
        ('PATH', r'(?<![\w.])/(?:home|Users|data|mnt|root)/[^\s\x22\x27<>]+'),
        ('PATH', r'[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\s\x22\x27<>]+'),
        ('URL_AUTH', r'https?://[^\s/@:]+:[^\s/@]+@[^\s]+'),
    ]
    mapping, counts = {}, Counter()
    result = {'messages': []}
    for message in state['messages']:
        content = message['content']
        for category, pattern in patterns:
            def substitute(match):
                key = (category, match.group())
                if key not in mapping:
                    counts[category] += 1
                    mapping[key] = f'[REDACTED_{category}_{counts[category]}]'
                return mapping[key]
            content = re.sub(pattern, substitute, content)
        result['messages'].append({'role': message['role'], 'content': content})
    if result['messages'][-1]['role'] != 'user':
        raise ValueError('Future reply or invalid prefix boundary')
    return result, dict(counts)


def parse_json_text(text):
    """Accept one object, optionally inside one complete Markdown fence."""
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text, re.I)
        if not match:
            raise ValueError('invalid_json_fence')
        text = match.group(1)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    def constant(_):
        raise ValueError('nonfinite_json')
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        canonical(value)  # Also rejects floats that overflow, e.g. 1e999.
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid_json') from exc
    if not isinstance(value, dict):
        raise ValueError('json_object_required')
    return value


def parse_json_response(response):
    if not isinstance(response, dict) or response.get('stop_reason') in ('max_tokens', 'length'):
        raise ValueError('truncated_response')
    text = '\n'.join(b.get('text','') for b in response.get('content',[]) if b.get('type')=='text').strip()
    return parse_json_text(text)


def prepare_prefixes(splits, limit, split_names, audit=None):
    manifest = json.loads((splits/'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        from pathlib import Path
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid manifest path')
        if hashlib.sha256((splits/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen split artifact changed')
    parents = {r['conversation_id']:r for name in split_names for r in rows(splits/f'{name}.jsonl')}
    buckets = defaultdict(list)
    for row in rows(splits/'prefixes.jsonl'):
        if row['split'] not in split_names:
            continue
        parent = parents[row['conversation_id']]
        if row['group_id']!=parent['group_id']:
            raise ValueError('Prefix group mismatch')
        state, counts = sanitize_state(row['state'])
        # Credential-like inputs are excluded, not merely masked and sent onward.
        if any(k in counts for k in ('TOKEN','CREDENTIAL','URL_AUTH')):
            if audit is not None:
                audit.setdefault('privacy_exclusions',[]).append({'prefix_id':row['id'],'split':row['split'],
                    'reason_codes':sorted(k for k in counts if k in ('TOKEN','CREDENTIAL','URL_AUTH'))})
            continue
        buckets[parent['source']].append({
            'id':row['id'],'group_id':row['group_id'],'split':row['split'],
            'source':parent['source'],'source_family':parent['source_family'],
            'state':state,'original_prefix_sha256':row['prefix_sha256'],
            'redaction_counts':counts,'teacher_export_approved':True})
    for bucket in buckets.values():
        bucket.sort(key=lambda r:digest([20260920,r['id']]))
    selected=[]
    while any(buckets.values()) and len(selected)<limit:
        for source in sorted(buckets):
            if buckets[source] and len(selected)<limit:
                selected.append(buckets[source].pop(0))
    return selected
