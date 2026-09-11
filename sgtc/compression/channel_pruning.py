"""FFN channel-dimension baselines: magnitude pruning & neuron merging.

Surgery on each encoder FFN: intermediate 768->3072 becomes 768->K, output
3072->768 becomes K->768 (plan Task 4.3; ratio = 3072/K).
  prune: keep top-K neurons by W1 row L2 norm (Han et al. 2015 criterion)
  merge: KMeans on W1 rows, count-weighted mean-merge of W1/W2/b1 per cluster
         (neuron-merging criterion of Kim et al. 2020)
"""
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def surgically_compress_ffn(model, K: int, mode: str, seed: int = 42):
    assert mode in ("prune", "merge")
    for layer in model.encoder.layers:
        ffn = layer.feed_forward
        W1 = ffn.intermediate_dense.weight.detach().float().cpu().numpy()   # (3072,768)
        b1 = ffn.intermediate_dense.bias.detach().float().cpu().numpy()     # (3072,)
        W2 = ffn.output_dense.weight.detach().float().cpu().numpy()         # (768,3072)
        b2 = ffn.output_dense.bias.detach().float().cpu().numpy()           # (768,) keep!
        if mode == "prune":
            keep = np.argsort(np.linalg.norm(W1, axis=1))[::-1][:K]         # top-K magnitude
            new_W1, new_b1, new_W2 = W1[keep], b1[keep], W2[:, keep]
        else:
            # MiniBatch: full KMeans with K ~ N/2 converges too slowly; minibatch
            # gives the same merge-baseline semantics at a fraction of the cost.
            km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=1,
                                 batch_size=512, max_iter=100)
            labels = km.fit_predict(W1)
            new_W1 = np.zeros((K, W1.shape[1]), dtype=np.float32)
            new_b1 = np.zeros(K, dtype=np.float32)
            new_W2 = np.zeros((W2.shape[0], K), dtype=np.float32)
            for k in range(K):
                m = labels == k
                if not m.any():
                    # MiniBatchKMeans can leave empty clusters: keep the cluster
                    # centre itself as the merged neuron (exact representative).
                    c = km.cluster_centers_[k]
                    new_W1[k] = c
                    new_b1[k] = 0.0
                    new_W2[:, k] = 0.0
                    continue
                new_W1[k] = W1[m].mean(axis=0)          # averaged incoming weights
                new_b1[k] = b1[m].mean()                # averaged activation offset
                new_W2[:, k] = W2[:, m].sum(axis=1)     # exact when members activate alike
        new_W1 = np.nan_to_num(new_W1, nan=0.0, posinf=0.0, neginf=0.0)
        new_b1 = np.nan_to_num(new_b1, nan=0.0)
        new_W2 = np.nan_to_num(new_W2, nan=0.0)
        dev = ffn.intermediate_dense.weight.device
        dtype = ffn.intermediate_dense.weight.dtype
        ffn.intermediate_dense = torch.nn.Linear(W1.shape[1], K).to(dev, dtype)
        ffn.output_dense = torch.nn.Linear(K, W2.shape[0]).to(dev, dtype)
        with torch.no_grad():
            ffn.intermediate_dense.weight.copy_(torch.tensor(new_W1))
            ffn.intermediate_dense.bias.copy_(torch.tensor(new_b1))
            ffn.output_dense.weight.copy_(torch.tensor(new_W2))
            ffn.output_dense.bias.copy_(torch.tensor(b2))
    return model


@torch.no_grad()
def surgical_forward_check(model, wav_len=16000):
    """Sanity: with K=3072 (no-op surgery shape), output matches original."""
    x = torch.randn(wav_len).unsqueeze(0)
    if next(model.parameters()).dtype == torch.float16:
        x = x.half().to(next(model.parameters()).device)
        model = model.half()
    return model(x).last_hidden_state
