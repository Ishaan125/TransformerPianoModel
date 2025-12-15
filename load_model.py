import torch
import numpy as np
from model import TransformerNextNoteModel

def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu', override_hparams=None):
    ckpt = torch.load(checkpoint_path, map_location=device)
    hparams = ckpt.get('hparams', {})
    if override_hparams:
        hparams.update(override_hparams)
    vocab_size = hparams.get('vocab_size')
    d_model = hparams.get('d_model', 256)
    nhead = hparams.get('nhead', 8)
    nlayers = hparams.get('nlayers', 4)
    model = TransformerNextNoteModel(vocab_size=vocab_size, d_model=d_model, nhead=nhead, nlayers=nlayers)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model

def sample_next(model, seq_tokens, seq_vel=None, seq_dur=None, device='cuda' if torch.cuda.is_available() else 'cpu', temperature=1.0, top_k=10, repeat_penalty: float = 1.0):
    # seq_tokens: torch.LongTensor shape (1, seq_len)
    with torch.no_grad():
        logits, vel_pred, dur_pred = model(seq_tokens.to(device),
                                           velocities=(seq_vel.to(device) if seq_vel is not None else None),
                                           durations=(seq_dur.to(device) if seq_dur is not None else None))
        logits = logits.squeeze(0)  # (vocab,)
        # discourage repeating the immediately previous token by applying a simple
        # repetition penalty to its logit (make it less likely to be sampled)
        # also discourage the same sequence of 2 notes repeatedly
        if repeat_penalty is not None and repeat_penalty > 1.0:
            try:
                prev_token = int(seq_tokens[0, -1].item())
                if 0 <= prev_token < logits.size(0):
                    logits[prev_token] = logits[prev_token] / float(repeat_penalty)
                second_prev_token = int(seq_tokens[0, -2].item())
                if 0 <= second_prev_token < logits.size(0):
                    logits[second_prev_token] = logits[second_prev_token] / float(repeat_penalty)
            except Exception:
                # if seq_tokens isn't as expected or its currently on the first or second token, skip penalty
                pass
        # apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        # top-k sampling
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, top_k)
            probs = torch.softmax(topk_vals, dim=-1)
            idx = topk_idx[torch.multinomial(probs, num_samples=1)]
            next_token = idx.item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token, vel_pred.item() if vel_pred is not None else None, dur_pred.item() if dur_pred is not None else None

def autoregressive_generate(model, seed_tokens, gen_steps=100, temperature=1.0, top_k=10, device='cuda' if torch.cuda.is_available() else 'cpu', repeat_penalty: float = 1.0):
    seq = list(seed_tokens)
    generated = []
    for _ in range(gen_steps):
        inp = torch.LongTensor([seq[-len(seed_tokens):]])  # (1, seq_len)
        next_token, next_vel, next_dur = sample_next(model, inp, device=device, temperature=temperature, top_k=top_k, repeat_penalty=repeat_penalty)
        generated.append((next_token, next_vel, next_dur))
        seq.append(next_token)
    return generated