"""Task 4.4: SGTC-deploy -- in-encoder temporal merging (deployment variant).

Merging happens INSIDE the encoder after layer `merge_after` (default 1) so the
remaining layers process T/ratio tokens (real wall-clock savings).  The
sensitivity proxy is adjacent-frame cosine similarity (cheap, no head calls):
the most-similar adjacent pairs merge first, mirroring SGTC's observation that
low-sensitivity frames are near-duplicates of their neighbours (paper II.D
approximation strategy).

Equivalence guard: with merge disabled, deploy_forward reproduces
model(x).last_hidden_state (fp16 tolerance).
"""
import numpy as np
import torch


def merge_adjacent_rounds(h: torch.Tensor, T_target: int) -> torch.Tensor:
    """h: (1, T, N). Round-based merging: each round greedily selects the
    most-similar DISJOINT adjacent pairs (vectorised selection, one cat per
    round), so ~log(T/T') rounds instead of T/2 python iterations."""
    while h.shape[1] > T_target:
        a, b = h[:, :-1], h[:, 1:]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)[0].cpu().numpy()
        order = np.argsort(-cos, kind="stable")
        chosen, blocked = [], set()
        for i in order:
            if i - 1 in blocked or i in blocked or i + 1 in blocked:
                continue
            chosen.append(int(i))
            blocked.update([i - 1, i, i + 1])
        if not chosen:
            break
        new = (h[:, chosen] + h[:, list(np.array(chosen) + 1)]) / 2.0   # (1, C, N)
        drop = set(chosen) | {i + 1 for i in chosen}
        keep = [t for t in range(h.shape[1]) if t not in drop]
        h = torch.cat([h[:, keep], new], dim=1)
        if h.shape[1] < T_target:                     # merged slightly too far
            break
    return h


@torch.no_grad()
def deploy_forward(model, input_values: torch.Tensor, ratio: int = 2,
                   merge_after: int = 1, disable_merge: bool = False):
    """Manual w2v2 forward with in-encoder merging; returns last_hidden_state."""
    extract = model.feature_extractor(input_values)
    proj = model.feature_projection(extract.transpose(1, 2))
    hidden = proj[0] if isinstance(proj, tuple) else proj
    enc = model.encoder
    hidden = hidden + enc.pos_conv_embed(hidden)     # residual positional conv
    hidden = enc.layer_norm(hidden)
    for i, layer in enumerate(enc.layers):
        hidden = layer(hidden, attention_mask=None, output_attentions=False)[0]
        if not disable_merge and i == merge_after and hidden.shape[1] > hidden.shape[1] // ratio:
            hidden = merge_adjacent_rounds(hidden, hidden.shape[1] // ratio)
    return hidden                                    # encoder applies no final norm
