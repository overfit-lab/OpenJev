"""Freeze grouped splits after a complete lexical join within the selected pool."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

try:
    from .prepare_chat_pool import Groups, canonical, digest, document_anchor, normalize, read_at, write_json, write_jsonl
except ImportError:
    from prepare_chat_pool import Groups, canonical, digest, document_anchor, normalize, read_at, write_json, write_jsonl


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def threshold_join(sets, threshold):
    """Exact Jaccard join using globally ordered prefix filtering, no LSH recall loss."""
    frequency = Counter(token for tokens in sets for token in tokens)
    ordered = [sorted(tokens, key=lambda t: (frequency[t], t)) for tokens in sets]
    postings = defaultdict(list)
    for index in sorted(range(len(sets)), key=lambda i: (len(sets[i]), i)):
        tokens = ordered[index]
        if not tokens:
            continue
        prefix = tokens[:len(tokens) - math.ceil(threshold * len(tokens)) + 1]
        candidates = {other for token in prefix for other in postings[token]
                      if len(sets[other]) >= threshold * len(tokens)}
        for other in sorted(candidates):
            overlap = len(sets[index] & sets[other])
            score = overlap / (len(sets[index]) + len(sets[other]) - overlap)
            if score >= threshold:
                yield other, index, score
        for token in prefix:
            postings[token].append(index)


def allocate(components, config):
    fractions = config['fractions']
    size = len(components)
    counts = {split: math.floor(size * fraction) for split, fraction in fractions.items()}
    remaining = size - sum(counts.values())
    order = sorted(fractions, key=lambda s: (-(size * fractions[s] - counts[s]), s))
    for split in order[:remaining]:
        counts[split] += 1
    assigned, actual = {}, Counter()
    for group, item in components.items():
        if config['holdout_family'] in item['families']:
            assigned[group] = config['holdout_split']
            actual[config['holdout_split']] += 1
    if actual[config['holdout_split']] > counts[config['holdout_split']]:
        raise ValueError('Holdout exceeds test budget; version the allocation configuration')
    pending = sorted((g for g in components if g not in assigned),
                     key=lambda g: digest([config['seed'], g]))
    for group in pending:
        split = max(fractions, key=lambda s: (counts[s] - actual[s], fractions[s]))
        assigned[group] = split
        actual[split] += 1
    return assigned, counts


def freeze(root, pool, output, config):
    if output.exists():
        raise ValueError('Output exists; use a new directory or archive the old run')
    conversations = list(rows(pool / 'conversations.jsonl'))
    members = list(rows(pool / 'selected_group_members.jsonl'))
    quarantine = list(rows(pool / 'exploration_quarantine.jsonl'))
    source_config = json.loads((pool / 'build_config.json').read_text())
    sources = {s['id']: s['upstream_family_hint'] for s in source_config['sources']}
    groups, node_for, keys, views = Groups(), {}, {}, []
    for group in sorted({r['group_id'] for r in conversations + quarantine}):
        node_for[group] = groups.add(any(q['group_id'] == group for q in quarantine))

    def add_view(text, node):
        value = normalize(text)
        if len(value) < config['min_view_characters']:
            return
        key = hashlib.sha256(value.encode()).hexdigest()
        if key in keys:
            groups.union(node, keys[key])
            return
        keys[key] = node
        tokens = {int.from_bytes(hashlib.blake2b(value[i:i+5].encode(), digest_size=8).digest(), 'little')
                  for i in range(len(value)-4)}
        views.append({'node': node, 'tokens': tokens, 'sha256': key})

    def add_messages(messages, node):
        users = [m['content'] for m in messages if m['role'] == 'user']
        add_view('\n'.join(users), node)
        for content in users:
            add_view(content, node)
            anchor = document_anchor(content)
            if anchor:
                add_view(anchor, node)

    for entry in members:
        row = read_at(root, entry)
        add_messages(row['messages'], node_for[entry['group_id']])
    recipes = json.loads(Path(config.get('exclusion_anchors_file', 'configs/data/source_exclusions.json')).read_text())
    for item in quarantine:
        recipe = next(r for r in recipes if r['file'] == item['source_file'] and r['expected_source_id'] == item['source_id'])
        with (root / recipe['file']).open('rb') as stream:
            if 'byte_offset' in recipe:
                stream.seek(recipe['byte_offset'])
                raw = stream.readline()
                if hashlib.sha256(raw).hexdigest() != recipe['raw_sha256']:
                    raise ValueError('Exclusion anchor changed')
            else:
                for _ in range(recipe['line']):
                    raw = stream.readline()
        row = json.loads(raw)
        if row['id'] != item['source_id']:
            raise ValueError('Exploration source changed')
        add_messages(row['messages'], node_for[item['group_id']])
    print(f"Reviewing {len(views)} distinct user/document views", flush=True)
    edges = []
    for left, right, score in threshold_join([v['tokens'] for v in views], config['lexical_jaccard']):
        if groups.find(views[left]['node']) != groups.find(views[right]['node']):
            edges.append({'left_view_sha256': views[left]['sha256'], 'right_view_sha256': views[right]['sha256'], 'jaccard': score})
            groups.union(views[left]['node'], views[right]['node'])
    components = {}
    root_ids = defaultdict(list)
    for group, node in node_for.items():
        root_ids[groups.find(node)].append(group)
    reviewed = {group: 'rg_' + digest(sorted(root_ids[groups.find(node)])) for group, node in node_for.items()}
    excluded = set()
    for row in conversations:
        original = row['group_id']
        if groups.explored[groups.find(node_for[original])]:
            excluded.add(original)
            continue
        group = reviewed[original]
        components.setdefault(group, {'original_group_ids': set(), 'families': set()})['original_group_ids'].add(original)
    for entry in members:
        group = reviewed[entry['group_id']]
        if group in components:
            components[group]['families'].add(sources[entry['source']])
    assigned, counts = allocate(components, config)
    output.mkdir(parents=True)
    manifests = {name: [] for name in config['fractions']}
    for row in conversations:
        if row['group_id'] in excluded:
            continue
        group = reviewed[row['group_id']]
        split = assigned[group]
        manifests[split].append({'conversation_id': row['id'], 'original_group_id': row['group_id'],
            'group_id': group, 'split': split, 'source': row['provenance']['source'],
            'source_family': sources[row['provenance']['source']],
            'evaluation_slice': 'source_holdout' if config['holdout_family'] in components[group]['families'] else 'in_distribution',
            'source_raw_sha256': row['provenance']['raw_sha256']})
    lookup = {r['conversation_id']: r for batch in manifests.values() for r in batch}
    prefix_rows = []
    for row in rows(pool / 'prefixes.jsonl'):
        if row['conversation_id'] in lookup:
            parent = lookup[row['conversation_id']]
            prefix_rows.append({**row, 'original_group_id': row['group_id'], 'group_id': parent['group_id'],
                                'split': parent['split'], 'evaluation_slice': parent['evaluation_slice']})
    for split, batch in manifests.items():
        write_jsonl(output / f'{split}.jsonl', sorted(batch, key=lambda r: r['conversation_id']))
    write_jsonl(output / 'prefixes.jsonl', prefix_rows)
    write_jsonl(output / 'reviewed_groups.jsonl', ({'group_id': g, 'split': assigned[g],
               'original_group_ids': sorted(c['original_group_ids']), 'source_families': sorted(c['families'])}
               for g, c in sorted(components.items())))
    write_jsonl(output / 'member_assignments.jsonl', ({**r, 'original_group_id': r['group_id'],
               'group_id': reviewed[r['group_id']], 'split': assigned.get(reviewed[r['group_id']], 'exploration_quarantine')}
               for r in members))
    write_jsonl(output / 'lexical_review_edges.jsonl', edges)
    write_json(output / 'build_config.json', config)
    summary = {'version': config['version'], 'source_conversations': len(conversations), 'source_member_records': len(members),
        'reviewed_views': len(views), 'new_component_merge_edges': len(edges), 'excluded_original_groups': len(excluded),
        'assigned_groups': len(components), 'assigned_conversations': len(lookup), 'assigned_prefixes': len(prefix_rows),
        'group_counts': counts, 'conversation_counts': {k: len(v) for k,v in manifests.items()},
        'prefix_counts': dict(Counter(r['split'] for r in prefix_rows)),
        'source_family_counts': {split: dict(Counter(r['source_family'] for r in batch)) for split,batch in manifests.items()},
        'source_holdout_family': config['holdout_family'], 'teacher_calls_before_freeze': 0,
        'scope': config['review_scope'], 'upstream_split_recovery': 'unresolved',
        'final_test_label_approval': False, 'pool_manifest_sha256': hashlib.sha256((pool/'manifest.json').read_bytes()).hexdigest()}
    write_json(output/'summary.json', summary)
    write_json(output/'manifest.json', {'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted(output.iterdir()) if p.is_file()}})
    print(canonical(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool', type=Path, default=Path('outputs/source_pool_new'))
    parser.add_argument('--output', type=Path, default=Path('outputs/source_splits_new'))
    parser.add_argument('--config', type=Path, default=Path('configs/data/source_splits.json'))
    args = parser.parse_args()
    freeze(args.root, args.pool, args.output, json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
