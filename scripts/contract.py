"""Shared public request, input compiler and response contract (stdlib)."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math

COMPILER_VERSION = 'jev-fields-v1'
CONFIDENCE_METHOD = 'top2_margin_v1'
BOUNDARIES = ['<|openjev_state|>', '<|openjev_question|>',
              '<|openjev_candidate|>', '<|openjev_decision|>']


def serialize(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).replace('<', '\\u003c')


def json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value: json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for item in value.values(): json_value(item)
        return
    raise ValueError('Expected finite JSON value')


def description(value, nullable=False):
    if nullable and value is None: return
    if not isinstance(value, (str, dict, list)) or not value:
        raise ValueError('Expected nonempty string, object or array')
    json_value(value)


def validate_request(request, supported=('choice', 'noul', 'score')):
    if not isinstance(request, dict) or set(request) - {'model', 'state', 'questions'}:
        raise ValueError('Unknown request fields')
    if 'state' not in request or 'questions' not in request:
        raise ValueError('state and questions are required')
    if not isinstance(request['state'], (str, dict, list)):
        raise ValueError('Invalid state')
    json_value(request['state'])
    if 'model' in request and (not isinstance(request['model'], str) or not request['model']):
        raise ValueError('Invalid model identifier')
    questions = request['questions']
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError('Expected 1–64 named questions')
    for key, q in questions.items():
        if not isinstance(key, str) or not key or not isinstance(q, dict):
            raise ValueError('Invalid question key/definition')
        kind = q.get('type')
        if kind not in supported:
            raise ValueError('Unsupported primitive')
        expected = {'type', 'instructions'} | ({'criteria'} if kind != 'noul' else set())
        if set(q) != expected: raise ValueError('Invalid question fields')
        description(q['instructions'])
        if kind == 'choice':
            criteria = q['criteria']
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError('Choice requires 2–255 criteria')
            for name, value in criteria.items():
                if not isinstance(name, str) or not name.strip(): raise ValueError('Invalid candidate name')
                description(value, nullable=True)
        elif kind == 'score':
            if not isinstance(q['criteria'], list) or not 2 <= len(q['criteria']) <= 10:
                raise ValueError('Score requires 2–10 levels')
            for value in q['criteria']: description(value)
    return request


@dataclass
class CompiledQuestion:
    question_id: str
    kind: str
    ids: list[str]
    paths: list[list[int]]


def compile_request(tokenizer, request, max_length=8192, max_total_tokens=262144):
    validate_request(request)
    tokens = tokenizer.convert_tokens_to_ids(BOUNDARIES)
    if len(set(tokens)) != 4 or any(x is None or x == getattr(tokenizer, 'unk_token_id', None) for x in tokens):
        raise ValueError('Register the four structural tokens before compilation')
    encode = lambda x: tokenizer.encode(serialize(x), add_special_tokens=False)
    state = [tokens[0]] + encode(request['state'])
    compiled = []
    total = 0
    for qid, q in request['questions'].items():
        base = state + [tokens[1]] + encode({'type': q['type'], 'instructions': q['instructions']})
        if q['type'] == 'choice':
            ids = sorted(q['criteria'])
            candidates = [{'name': name, 'description': q['criteria'][name]} for name in ids]
            base += encode({'criteria': candidates})
        elif q['type'] == 'score':
            ids = [str(i) for i in range(len(q['criteria']))]
            candidates = [{'description': value} for value in q['criteria']]
        else:
            ids = ['false', 'true']
            candidates = [{'answer': False, 'meaning': 'No'}, {'answer': True, 'meaning': 'Yes'}]
        paths = [base + [tokens[2]] + encode(candidate) + [tokens[3]] for candidate in candidates]
        if any(len(p) > max_length for p in paths):
            raise ValueError('Path exceeds context budget; no truncation')
        total += sum(map(len, paths))
        if total > max_total_tokens: raise ValueError('Request exceeds total compiled token budget')
        compiled.append(CompiledQuestion(qid, q['type'], ids, paths))
    return compiled


def response_from_logits(request, logits, model, temperature=1.0, tokenizer=None, input_tokens=None):
    validate_request(request)
    if not isinstance(model, str) or not model: raise ValueError('Missing output model id')
    if not math.isfinite(temperature) or temperature <= 0: raise ValueError('Invalid temperature')
    if set(logits) != set(request['questions']): raise ValueError('Logit question mapping mismatch')
    answers = {}
    for qid, q in request['questions'].items():
        ids = (sorted(q['criteria']) if q['type'] == 'choice' else
               [str(i) for i in range(len(q['criteria']))] if q['type'] == 'score' else ['false', 'true'])
        z = logits[qid]
        if len(z) != len(ids) or any(type(v) not in (float, int) or not math.isfinite(v) for v in z):
            raise ValueError('Invalid candidate logits')
        values = [math.exp((v - max(z)) / temperature) for v in z]
        probabilities = [v / sum(values) for v in values]
        if q['type'] == 'choice':
            ordered = sorted(probabilities, reverse=True)
            answers[qid] = {'type': 'choice', 'choice': ids[probabilities.index(max(probabilities))],
                            'confidence': ordered[0] - ordered[1],
                            'probabilities': dict(zip(ids, probabilities))}
        elif q['type'] == 'score':
            ordered = sorted(probabilities, reverse=True)
            answers[qid] = {'type': 'score', 'score': sum(i * p for i, p in enumerate(probabilities)),
                            'legend': dict(zip(ids, q['criteria'])),
                            'probabilities': dict(zip(ids, probabilities)),
                            'confidence': ordered[0] - ordered[1]}
        else:
            answers[qid] = {'type': 'noul', 'noul': probabilities[1]}
    result = {'model': model, 'answers': answers}
    if tokenizer is not None:
        if type(input_tokens) is not int or input_tokens < 0: raise ValueError('Missing actual input token count')
        result['usage'] = {'input_tokens': input_tokens,
                           'output_tokens': len(tokenizer.encode(serialize(answers), add_special_tokens=False))}
    return result
