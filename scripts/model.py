"""Backbone-independent candidate scorer and self-contained PEFT checkpoints."""
from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import save_file, load_file
try:
    from .contract import BOUNDARIES, COMPILER_VERSION, CONFIDENCE_METHOD, compile_request, response_from_logits
except ImportError:
    from contract import BOUNDARIES, COMPILER_VERSION, CONFIDENCE_METHOD, compile_request, response_from_logits


class BoundaryEmbedding(nn.Module):
    def __init__(self, base, ids):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.register_buffer('boundary_ids', torch.tensor(ids, device=base.weight.device))
        self.new_rows = nn.Parameter(base.weight.detach().float().mean(0).to(base.weight.dtype).repeat(len(ids), 1))
    @property
    def weight(self): return self.base.weight
    def forward(self, ids):
        output = self.base(ids.clamp_max(self.base.num_embeddings - 1))
        for i, token in enumerate(self.boundary_ids):
            output = torch.where((ids == token).unsqueeze(-1), self.new_rows[i], output)
        return output


class CandidateScorer(nn.Module):
    def __init__(self, backbone, head, pad_id):
        super().__init__()
        self.backbone, self.head, self.pad_id = backbone, head, pad_id
    def _chunk(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        index = attention_mask.sum(-1) - 1
        readout = hidden[torch.arange(hidden.shape[0], device=hidden.device), index]
        return self.head(readout.to(self.head.weight.dtype)).squeeze(-1).float()
    def forward(self, paths, chunk_size=4, recompute=True):
        if not paths or chunk_size < 1: raise ValueError('Nonempty paths and positive chunk size required')
        device = self.head.weight.device; outputs = []
        for start in range(0, len(paths), chunk_size):
            batch = paths[start:start + chunk_size]; width = max(map(len, batch))
            ids = torch.full((len(batch), width), self.pad_id, device=device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, path in enumerate(batch):
                ids[i, :len(path)] = torch.tensor(path, device=device); mask[i, :len(path)] = 1
            outputs.append(checkpoint(self._chunk, ids, mask, use_reentrant=False, preserve_rng_state=True)
                           if self.training and recompute else self._chunk(ids, mask))
        return torch.cat(outputs)


def public_identity(model_dir, model_id=None, revision=None):
    path = Path(model_dir) / 'openjev_revision.json'
    recorded = json.loads(path.read_text()) if path.exists() else {}
    identity = {'model': model_id or recorded.get('repo'), 'revision': revision or recorded.get('revision')}
    for key, value in identity.items():
        if not isinstance(value, str) or not value or value.startswith(('/', '.')) or '\\' in value or '://' in value:
            raise ValueError('Provide public model identity and pinned revision, not local paths')
    return identity


def load_reference(model_dir, device, dtype, identity):
    base = AutoModel.from_pretrained(model_dir, local_files_only=True, dtype=dtype, attn_implementation='sdpa')
    if not hasattr(base.config, 'hidden_size'): raise ValueError('Backbone must expose hidden_size')
    base.config.use_cache = False; base.config._name_or_path = identity['model']
    return base.to(device)


def build(base, tokenizer, method, config, seed, identity):
    rows = BoundaryEmbedding(base.get_input_embeddings(), tokenizer.convert_tokens_to_ids(BOUNDARIES))
    base.set_input_embeddings(rows)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed + 17)
        backbone = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
            init_lora_weights='pissa_niter_4' if method == 'pissa' else True, **config['adapter']))
    backbone.peft_config['default'].base_model_name_or_path = identity['model']
    rows.new_rows.requires_grad_(True)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        head = nn.Linear(base.config.hidden_size, 1, bias=False, device=rows.weight.device, dtype=torch.float32)
    return CandidateScorer(backbone, head, tokenizer.pad_token_id), rows


def save_checkpoint(model, rows, tokenizer, destination, metadata):
    destination = Path(destination)
    if destination.exists(): raise ValueError('Checkpoint exists; choose a new path')
    destination.mkdir(parents=True)
    model.backbone.save_pretrained(destination / 'adapter', save_embedding_layers=False)
    config_path = destination / 'adapter/adapter_config.json'
    config = json.loads(config_path.read_text())
    config['init_lora_weights'] = True  # residual base is already decomposed
    config['base_model_name_or_path'] = metadata['model']
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    base = model.backbone.get_base_model()
    embedding_name = next(name for name, module in base.named_modules() if module is rows)
    state = {}
    for name, tensor in base.state_dict().items():
        if '.lora_' in name or name in (embedding_name + '.new_rows', embedding_name + '.boundary_ids'): continue
        name = name.replace('.base_layer.', '.')
        if name.startswith(embedding_name + '.base.'):
            name = embedding_name + name[len(embedding_name + '.base'):]
        state[name] = tensor.detach().cpu().contiguous().clone()
    residual = destination / 'residual_base'; residual.mkdir()
    save_file(state, residual / 'model.safetensors', metadata={'format': 'pt'})
    base.config._name_or_path = metadata['model']; base.config.save_pretrained(residual)
    save_file({'head': model.head.weight.detach().cpu().contiguous(),
               'boundary_rows': rows.new_rows.detach().cpu().contiguous()}, destination / 'readout.safetensors')
    tokenizer.name_or_path = metadata['model']; tokenizer.init_kwargs['name_or_path'] = metadata['model']
    tokenizer.save_pretrained(destination / 'tokenizer')
    metadata = {**metadata, 'compiler_version': COMPILER_VERSION, 'confidence_method': CONFIDENCE_METHOD,
                'boundary_ids': tokenizer.convert_tokens_to_ids(BOUNDARIES),
                'format': 'residual_base_plus_adapter', 'architecture': 'B',
                'usage_method': 'compiled_paths_and_serialized_answers_v1'}
    (destination / 'openjev.json').write_text(json.dumps(metadata, indent=2) + '\n')


def load_checkpoint(path, device='cpu', dtype=torch.float32, trainable=False):
    path = Path(path); metadata = json.loads((path / 'openjev.json').read_text())
    if metadata['compiler_version'] != COMPILER_VERSION: raise ValueError('Unsupported checkpoint compiler')
    base = load_reference(path / 'residual_base', device, dtype, metadata)
    tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer', local_files_only=True)
    rows = BoundaryEmbedding(base.get_input_embeddings(), metadata['boundary_ids'])
    base.set_input_embeddings(rows)
    tensors = load_file(str(path / 'readout.safetensors'))
    with torch.no_grad(): rows.new_rows.copy_(tensors['boundary_rows'].to(device))
    backbone = PeftModel.from_pretrained(base, path / 'adapter', is_trainable=trainable)
    rows.new_rows.requires_grad_(trainable)
    head = nn.Linear(base.config.hidden_size, 1, bias=False, device=device, dtype=torch.float32)
    with torch.no_grad(): head.weight.copy_(tensors['head'].to(device))
    scorer = CandidateScorer(backbone, head, tokenizer.pad_token_id)
    scorer.train(trainable)
    return scorer, rows, tokenizer, metadata


@torch.no_grad()
def predict(model, tokenizer, metadata, request, chunk_size=2):
    model.eval()
    requested = request.get('model') if isinstance(request, dict) else None
    if requested is not None and requested not in (metadata.get('output_model', 'openjev-preview'), 'openjev-latest'):
        raise ValueError('Unknown local model identifier')
    compiled = compile_request(tokenizer, request, max_length=metadata.get('max_length',8192),
                               max_total_tokens=metadata.get('max_total_tokens',262144))
    logits = {q.question_id: model(q.paths, chunk_size, recompute=False).cpu().tolist() for q in compiled}
    return response_from_logits(request, logits, metadata.get('output_model','openjev-preview'),
        temperature=metadata.get('temperature',1.0), tokenizer=tokenizer,
        input_tokens=sum(len(p) for q in compiled for p in q.paths))
