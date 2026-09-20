"""Prepare a local, traceable conversation pool; no teacher calls or split fitting.

Exact groups are built over every valid record in the registered sources.
Approximate lexical grouping is explicitly limited to reservoir representatives.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def valid_messages(record):
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing_messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("unsupported_role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("empty_or_nontext_message")
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("missing_user_turn")
    return messages


def document_anchor(text):
    """Conservative visible-document extraction, not a semantic label extractor."""
    text = unicodedata.normalize("NFKC", text)
    if "\n问题:" in text:
        article = text.rsplit("\n问题:", 1)[0]
        article = re.sub(r"^阅读[^\n]*\n", "", article)
    elif "前提:" in text and "\n假设:" in text:
        article = text.split("前提:", 1)[1].split("\n假设:", 1)[0]
    else:
        before, separator, last = text.rpartition("\n")
        article = before if separator and ("?" in last or "？" in last) and len(before) >= 160 else ""
    article = normalize(article)
    return article if len(article) >= 80 else None


def exact_keys(messages):
    users = [m["content"] for m in messages if m["role"] == "user"]
    normalized = [normalize(value) for value in users]
    keys = {"users:" + digest(normalized)}
    # Shared substantive first prompts connect conversation branches.
    if len(normalized[0]) >= 80:
        keys.add("first:" + digest(normalized[0]))
    for text in users:
        anchor = document_anchor(text)
        if anchor:
            keys.add("document:" + digest(anchor))
    return keys


class Groups:
    def __init__(self):
        self.parent = []
        self.explored = []

    def add(self, explored=False):
        index = len(self.parent)
        self.parent.append(index)
        self.explored.append(explored)
        return index

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        low, high = sorted((left, right))
        self.parent[high] = low
        self.explored[low] = self.explored[low] or self.explored[high]


def prefix_indices(messages, max_characters):
    length, eligible = 0, []
    for index, message in enumerate(messages):
        length += len(message["content"])
        if length > max_characters:
            break
        if message["role"] == "user":
            eligible.append(index)
    return sorted({eligible[0], eligible[-1]}) if eligible else []


def privacy_flags(messages):
    text = "\n".join(m["content"] for m in messages)
    patterns = {
        "credential_pattern": r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{12,})\b",
        "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "ip_address": r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])",
        "mobile_number": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def script_profile(messages):
    text = "".join(m["content"] for m in messages if m["role"] == "user")
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk == 0:
        return "no_cjk_not_necessarily_english"
    return "cjk_dominant" if cjk >= latin else "mixed_latin_dominant"


def shingle_hashes(messages):
    text = normalize("\n".join(m["content"] for m in messages if m["role"] == "user"))
    if len(text) < 5:
        return {int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")}
    # All user text is fingerprinted, including long contexts; no silent truncation.
    return {int.from_bytes(hashlib.blake2b(text[i:i+5].encode(), digest_size=8).digest(), "little")
            for i in range(len(text) - 4)}


def jaccard(left, right):
    return len(left & right) / len(left | right) if left or right else 1.0


def near_groups(candidates, groups, config):
    import numpy as np
    rng = np.random.default_rng(config["seed"])
    count = config["minhash_permutations"]
    bands = config["minhash_bands"]
    if count % bands:
        raise ValueError("Minhash permutations must divide into bands")
    a = rng.integers(1, 2**63, size=count, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    b = rng.integers(0, 2**64 - 1, size=count, dtype=np.uint64)
    buckets, shingles, edges, considered = defaultdict(list), [], [], set()
    threshold = config["near_duplicate_jaccard"]
    for index, candidate in enumerate(candidates):
        hashes = shingle_hashes(candidate["messages"])
        shingles.append(hashes)
        values = np.array(sorted(hashes), dtype=np.uint64)
        signature = np.full(count, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, len(values), 2048):
            transformed = a[:, None] * values[None, start:start+2048] + b[:, None]
            signature = np.minimum(signature, transformed.min(axis=1))
        width = count // bands
        neighbors = set()
        for band in range(bands):
            key = (band, signature[band*width:(band+1)*width].tobytes())
            neighbors.update(buckets[key])
            buckets[key].append(index)
        for other in sorted(neighbors):
            considered.add((other, index))
            score = jaccard(hashes, shingles[other])
            if score >= threshold:
                groups.union(candidate["node"], candidates[other]["node"])
                edges.append({"left": candidate["node"], "right": candidates[other]["node"], "jaccard": score})
        if (index + 1) % 1000 == 0:
            print(f"Near-duplicate check: {index+1}/{len(candidates)} representatives", flush=True)
    return edges, len(considered)


def read_at(root, entry):
    with (root / entry["file"]).open("rb") as stream:
        stream.seek(entry["byte_offset"])
        raw = stream.readline()
    if hashlib.sha256(raw).hexdigest() != entry["raw_sha256"]:
        raise ValueError("Source row changed since scan")
    return json.loads(raw)


def prepare(root, output, config, exploration_recipes):
    output.mkdir(parents=True, exist_ok=False)
    groups, owners, catalog, explorers = Groups(), {}, [], []
    def register(messages, explored=False):
        node = groups.add(explored)
        for key in exact_keys(messages):
            if key in owners:
                groups.union(node, owners[key])
            else:
                owners[key] = node
        return node
    # Every already-inspected preview conversation is a quarantine anchor.
    seen_explorers = set()
    for recipe in exploration_recipes:
        locator = (recipe["file"], recipe["line"])
        if locator in seen_explorers:
            continue
        seen_explorers.add(locator)
        with (root / recipe["file"]).open("rb") as stream:
            if "byte_offset" in recipe:
                stream.seek(recipe["byte_offset"])
                raw = stream.readline()
                if hashlib.sha256(raw).hexdigest() != recipe["raw_sha256"]:
                    raise ValueError("Registered exclusion anchor changed")
            else:
                for _ in range(recipe["line"]):
                    raw = stream.readline()
        row = json.loads(raw)
        if row.get("id") != recipe["expected_source_id"]:
            raise ValueError("Exploration source changed")
        messages = valid_messages(row)
        node = register(messages, explored=True)
        catalog.append(None)
        explorers.append({"node": node, "messages": messages, "file": recipe["file"],
                          "source_id": row["id"], "raw_sha256": hashlib.sha256(raw).hexdigest()})
    source_entries, reservoirs = [], {}
    for source in config["sources"]:
        path = root / source["file"]
        sha, counts, heap = hashlib.sha256(), Counter(), []
        with path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                line_number += 1
                sha.update(raw)
                if not raw.strip():
                    continue
                counts["records"] += 1
                try:
                    row = json.loads(raw)
                    messages = valid_messages(row)
                except (ValueError, TypeError, AttributeError):
                    counts["invalid_records"] += 1
                    continue
                node = register(messages)
                raw_hash = hashlib.sha256(raw).hexdigest()
                entry = {
                    "node": node, "source": source["id"], "file": source["file"],
                    "line_1based": line_number, "byte_offset": offset, "raw_sha256": raw_hash,
                    "source_id": str(row.get("id", raw_hash)), "local_split": source["local_split"],
                }
                assert node == len(catalog)
                catalog.append(entry)
                counts["valid_records"] += 1
                indices = prefix_indices(messages, config["max_prefix_characters"])
                if not indices:
                    counts["no_prefix_within_budget"] += 1
                    continue
                if sum(len(m["content"]) for m in messages) > config["max_record_characters_for_sampling"]:
                    counts["over_record_character_budget"] += 1
                    continue
                flags = privacy_flags(messages)
                if "credential_pattern" in flags:
                    counts["credential_pattern_excluded_from_sampling"] += 1
                    continue
                counts["sampling_eligible_records"] += 1
                rank = int(digest({"seed": config["seed"], "source": source["id"], "raw_sha256": raw_hash}), 16)
                item = (-rank, node, {"node": node, "rank": rank, "source": source["id"],
                                     "messages": messages, "prefix_indices": indices, "privacy_flags": flags})
                limit = source["quota"] * config["reservoir_multiplier"]
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
        if sha.hexdigest() != source["expected_sha256"]:
            raise ValueError(f"Registered source SHA256 changed: {source['id']}")
        source_entries.append({**source, "counts": dict(counts), "actual_sha256": sha.hexdigest(),
                               "bytes": path.stat().st_size, "review_status": "local_preparation_only"})
        reservoirs[source["id"]] = sorted((item[2] for item in heap), key=lambda c: (c["rank"], c["node"]))
        print(f"Scanned {source['id']}: {counts['valid_records']} valid records", flush=True)
    exact_roots = {groups.find(i) for i, entry in enumerate(catalog) if entry}
    cross_file_groups = defaultdict(set)
    for entry in catalog:
        if entry:
            cross_file_groups[groups.find(entry["node"])].add(entry["file"])
    # Bounded pass: other branches in each exact group are not searched lexically.
    representatives = {groups.find(c["node"]): c for c in explorers}
    for source in config["sources"]:
        for candidate in reservoirs[source["id"]]:
            representatives.setdefault(groups.find(candidate["node"]), candidate)
    candidates = list(representatives.values())
    edges, comparisons = near_groups(candidates, groups, config)
    used, selected, selection_counts = set(), [], Counter()
    for source in config["sources"]:
        for candidate in reservoirs[source["id"]]:
            group = groups.find(candidate["node"])
            if group in used or groups.explored[group]:
                continue
            selected.append(candidate)
            used.add(group)
            selection_counts[source["id"]] += 1
            if selection_counts[source["id"]] == source["quota"]:
                break
        if selection_counts[source["id"]] != source["quota"] and not config.get("allow_source_shortfall", False):
            raise ValueError(f"Insufficient independent candidate groups for {source['id']}; increase a versioned reservoir budget")
    # Stable group identity uses the smallest member record hash.
    group_min = {}
    for explorer in explorers:
        group = groups.find(explorer["node"])
        group_min[group] = min(group_min.get(group, explorer["raw_sha256"]), explorer["raw_sha256"])
    for entry in catalog:
        if entry:
            group = groups.find(entry["node"])
            group_min[group] = min(group_min.get(group, entry["raw_sha256"]), entry["raw_sha256"])
    group_id = lambda node: "g_" + group_min[groups.find(node)]
    registry = {
        "version": config["version"], "sources": source_entries,
        "license_and_human_origin_verified": False, "original_upstream_splits_recovered": False,
    }
    write_json(output / "source_registry.json", registry)
    write_jsonl(output / "source_index.jsonl", (
        {**entry, "group_id": group_id(entry["node"]),
         "exploration_quarantine": groups.explored[groups.find(entry["node"])]}
        for entry in catalog if entry
    ))
    write_jsonl(output / "near_duplicate_edges.jsonl", edges)
    conversations, prefixes, selected_members = [], [], []
    group_sizes = Counter(groups.find(entry["node"]) for entry in catalog if entry)
    for candidate in selected:
        entry = catalog[candidate["node"]]
        reread = read_at(root, entry)
        if reread["messages"] != candidate["messages"]:
            raise ValueError("Conversation reconstruction failed")
        group = groups.find(candidate["node"])
        item = {
            "id": "conversation_" + entry["raw_sha256"], "group_id": group_id(candidate["node"]),
            "messages": candidate["messages"], "provenance": entry,
            "duplicate_group_records_in_scope": group_sizes[group],
            "candidate_user_message_indices": candidate["prefix_indices"],
            "script_profile": script_profile(candidate["messages"]), "privacy_flags": candidate["privacy_flags"],
            "split": None, "target": None, "training_eligible": False, "teacher_export_approved": False,
            "record_role": "source_conversation_not_model_input",
        }
        conversations.append(item)
        for index in candidate["prefix_indices"]:
            prefix = candidate["messages"][:index+1]
            prefixes.append({
                "id": item["id"] + f"/user-{index}", "group_id": item["group_id"],
                "conversation_id": item["id"], "user_message_index": index,
                "state": {"messages": prefix}, "prefix_sha256": digest(prefix),
                "privacy_flags": privacy_flags(prefix), "split": None, "target": None,
                "teacher_export_approved": False,
            })
    for entry in catalog:
        if entry and groups.find(entry["node"]) in used:
            selected_members.append({**entry, "group_id": group_id(entry["node"])})
    write_jsonl(output / "conversations.jsonl", conversations)
    write_jsonl(output / "prefixes.jsonl", prefixes)
    write_jsonl(output / "selected_group_members.jsonl", selected_members)
    write_jsonl(output / "exploration_quarantine.jsonl", (
        {"source_file": c["file"], "source_id": c["source_id"], "group_id": group_id(c["node"]),
         "reason": "Used in prior preview design/teacher pilot; exclude whole linked group from final test."}
        for c in explorers
    ))
    summary = {
        "version": config["version"], "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "scanned_sources": len(source_entries), "scanned_records": sum(s["counts"]["records"] for s in source_entries),
        "valid_records": sum(s["counts"]["valid_records"] for s in source_entries),
        "exact_groups_before_lexical_pass": len(exact_roots),
        "exact_cross_file_groups": sum(len(files) > 1 for files in cross_file_groups.values()),
        "duplicate_records_beyond_one_per_exact_group": sum(s["counts"]["valid_records"] for s in source_entries) - len(exact_roots),
        "near_pass_representatives": len(candidates), "near_candidate_pairs_checked": comparisons, "near_edges": len(edges),
        "exploration_anchors": len(explorers),
        "exploration_linked_records_in_scope": sum(groups.explored[groups.find(e["node"])] for e in catalog if e),
        "selected_conversations": len(conversations), "selected_groups": len(used), "selected_prefixes": len(prefixes),
        "selected_source_counts": dict(selection_counts),
        "selected_upstream_family_counts": dict(Counter(
            next(s["upstream_family_hint"] for s in config["sources"] if s["id"] == r["provenance"]["source"])
            for r in conversations)),
        "selected_script_profiles": dict(Counter(r["script_profile"] for r in conversations)),
        "selected_with_contact_flags": sum(bool(r["privacy_flags"]) for r in conversations),
        "selected_with_multiple_user_turns": sum(sum(m["role"] == "user" for m in r["messages"]) > 1 for r in conversations),
        "split_status": config["split_status"], "teacher_requests": 0, "labeled_decisions": 0,
        "config_sha256": digest(config), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            config["scope"], "Lexical LSH can miss near duplicates; no embedding or cross-language semantic grouping.",
            "Normalized grouping may conservatively merge distinct cases; it never transfers labels.",
            "User-role text is not proof of human authorship. Upstream licenses and original splits remain unresolved.",
            "Contact regex flags are incomplete. Raw pool is local only and not cleared for teacher export.",
        ],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "build_config.json", config)
    files = [p for p in output.iterdir() if p.is_file()]
    write_json(output / "manifest.json", {
        "version": config["version"], "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                                       "bytes": p.stat().st_size} for p in sorted(files)},
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset directory supplied at runtime; provenance paths are relative to it")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/source_pool.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/source_pool_new")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a fresh directory or move the old run to timestamped trash")
    config = json.loads(args.config.read_text())
    if config["max_prefixes_per_conversation"] != 2:
        parser.error("This version selects at most the first and last eligible user turn")
    recipes = json.loads((ROOT / "configs/data/source_exclusions.json").read_text())
    if sum(source["quota"] for source in config["sources"]) != config["target_groups"]:
        parser.error("Source quotas do not match target group count")
    result = prepare(args.root, args.output, config, recipes)
    print(f"Prepared {result['selected_groups']} groups and {result['selected_prefixes']} prefixes; splits remain unassigned.")


if __name__ == "__main__":
    main()
