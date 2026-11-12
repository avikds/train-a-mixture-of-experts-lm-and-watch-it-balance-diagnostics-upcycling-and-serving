"""
Train a Mixture-of-Experts LM and Watch It: Balance, Diagnostics, Upcycling and Serving

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - domain_chains
import numpy as np
import torch

def domain_chains(seed, n_domains=8, vocab=32, alpha=0.1):
    # Use a dedicated RNG so corpus generation is independent of model initialization.
    rng = np.random.default_rng([101, seed])

    # Generate one next-token probability table per domain.
    # Shape: (n_domains, vocab, vocab)
    P = rng.dirichlet(
        np.full(vocab, alpha),
        size=(n_domains, vocab)
    )

    # Return the probability tables as a float64 PyTorch tensor.
    return torch.tensor(P, dtype=torch.float64)

# Step 2 - sample_corpus
def stationary(P_d, iters=500):
    # Convert the transition matrix to a float64 NumPy array.
    if isinstance(P_d, torch.Tensor):
        P_d = P_d.detach().cpu().numpy()

    P_d = np.asarray(P_d, dtype=np.float64)

    V = P_d.shape[0]

    # Start with a uniform distribution over all vocabulary tokens.
    pi = np.full(V, 1.0 / V, dtype=np.float64)

    # Power iteration to approximate the stationary distribution.
    for _ in range(iters):
        pi = pi @ P_d

    # Normalize the resulting stationary distribution.
    return pi / pi.sum()


def sample_corpus(P, mix, n_seq, seq_len, seed):
    # Convert the transition tables to a float64 NumPy array.
    if isinstance(P, torch.Tensor):
        P_np = P.detach().cpu().numpy()
    else:
        P_np = np.asarray(P, dtype=np.float64)

    P_np = np.asarray(P_np, dtype=np.float64)

    # Get the number of domains and vocabulary size.
    D, V = P_np.shape[0], P_np.shape[1]

    # Convert mixture proportions to float64.
    mix = np.asarray(mix, dtype=np.float64)

    # Create a dedicated random number generator.
    rng = np.random.default_rng([202, seed])

    # Sample the domain for each sequence.
    domains = rng.choice(
        D,
        size=n_seq,
        p=mix / mix.sum()
    )

    # Initialize the token sequences.
    # Each sequence contains one tag token and seq_len regular tokens.
    tokens = np.empty(
        (n_seq, seq_len + 1),
        dtype=np.int64
    )

    # Column 0 contains the domain-specific tag token.
    tokens[:, 0] = V + domains

    # Calculate the stationary distribution for every domain.
    stationary_probs = np.stack([
        stationary(P_np[d]) for d in range(D)
    ])

    # Draw column 1 from the corresponding domain's stationary distribution.
    probs = stationary_probs[domains]
    cdf = np.cumsum(probs, axis=1)

    u = rng.random(n_seq)

    tokens[:, 1] = np.minimum(
        (u[:, None] > cdf).sum(axis=1),
        V - 1
    )

    # Generate the remaining tokens using domain-specific Markov transitions.
    for t in range(1, seq_len):
        # Select the transition probabilities based on the previous token.
        probs = P_np[domains, tokens[:, t]]

        # Calculate the cumulative distribution for each sequence.
        cdf = np.cumsum(probs, axis=1)

        # Generate one random number per sequence for this column.
        u = rng.random(n_seq)

        # Inverse-CDF sampling.
        tokens[:, t + 1] = np.minimum(
            (u[:, None] > cdf).sum(axis=1),
            V - 1
        )

    # Convert both outputs into int64 PyTorch tensors.
    return (
        torch.from_numpy(tokens),
        torch.from_numpy(domains)
    )

# Step 3 - entropy_floor
def entropy_floor(P, mix, seq_len):
    # Convert P to a float64 NumPy array.
    if isinstance(P, torch.Tensor):
        P = P.detach().cpu().numpy()

    P = np.asarray(P, dtype=np.float64)

    # Normalize the domain mixture weights.
    mix = np.asarray(mix, dtype=np.float64)
    weights = mix / mix.sum()

    n_domains = P.shape[0]
    domain_floors = np.zeros(n_domains, dtype=np.float64)

    for d in range(n_domains):
        # Get the stationary distribution for this domain.
        pi = stationary(P[d], iters=500)

        # Calculate the entropy of the stationary distribution.
        # Zero-probability terms contribute zero to entropy.
        positive_pi = pi > 0
        H_pi = -np.sum(
            pi[positive_pi] * np.log(pi[positive_pi])
        )

        # Calculate the entropy of every transition row.
        P_d = P[d]

        positive_P = P_d > 0
        log_terms = np.zeros_like(P_d, dtype=np.float64)

        log_terms[positive_P] = (
            P_d[positive_P] * np.log(P_d[positive_P])
        )

        row_entropies = -np.sum(log_terms, axis=1)

        # Expected transition entropy under the stationary distribution.
        expected_transition_entropy = np.sum(
            pi * row_entropies
        )

        # Calculate the average entropy floor for this domain.
        domain_floors[d] = (
            H_pi + (seq_len - 1) * expected_transition_entropy
        ) / seq_len

    # Calculate the mixture-weighted entropy floor.
    return float(np.sum(weights * domain_floors))

# Step 4 - DenseFFN
import torch.nn as nn
import torch.nn.functional as F


class DenseFFN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()

        # First weight matrix: (d, hidden)
        self.w1 = nn.Parameter(
            torch.randn(d, hidden) * d ** -0.5
        )

        # Second weight matrix: (hidden, d)
        self.w2 = nn.Parameter(
            torch.randn(hidden, d) * hidden ** -0.5
        )

    def forward(self, x):
        # Apply the first linear transformation, GELU, and the second transformation.
        return F.gelu(x @ self.w1) @ self.w2

# Step 5 - TinyLM
class Block(nn.Module):
    def __init__(self, d, n_heads, ffn):
        super().__init__()

        # Pre-normalization for attention.
        self.ln1 = nn.LayerNorm(d)

        # Combined query, key, and value projection.
        self.qkv = nn.Linear(d, 3 * d, bias=False)

        # Output projection after attention.
        self.proj = nn.Linear(d, d, bias=False)

        # Pre-normalization for the feed-forward module.
        self.ln2 = nn.LayerNorm(d)

        # Pluggable feed-forward module (DenseFFN or MoE).
        self.ffn = ffn

        self.n_heads = n_heads
        self.head_dim = d // n_heads

    def forward(self, x):
        B, T, d = x.shape

        # Normalize the input before attention.
        h = self.ln1(x)

        # Compute query, key, and value.
        q, k, v = self.qkv(h).split(d, dim=-1)

        # Reshape into multiple attention heads.
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply causal scaled dot-product attention.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True
        )

        # Merge attention heads back into the model dimension.
        attn = attn.transpose(1, 2).contiguous().reshape(B, T, d)

        # Attention output projection and residual connection.
        x = x + self.proj(attn)

        # Apply the feed-forward module with pre-normalization
        # and its residual connection.
        x = x + self.ffn(self.ln2(x))

        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d, n_layers, seq_len, make_ffn, n_heads=2):
        super().__init__()

        # Token embeddings.
        self.tok = nn.Embedding(vocab, d)

        # Positional embeddings.
        self.pos = nn.Embedding(seq_len, d)

        # Transformer blocks with a separately created FFN per layer.
        self.blocks = nn.ModuleList([
            Block(d, n_heads, make_ffn(i))
            for i in range(n_layers)
        ])

        # Final layer normalization.
        self.ln_f = nn.LayerNorm(d)

        # Output projection to vocabulary logits.
        self.head = nn.Linear(d, vocab, bias=False)

        self.seq_len = seq_len

    def forward(self, idx):
        B, T = idx.shape

        # Ensure the input sequence does not exceed the configured length.
        if T > self.seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds maximum length {self.seq_len}."
            )

        # Generate positional indices on the same device as the input.
        positions = torch.arange(T, device=idx.device)

        # Combine token and positional embeddings.
        x = self.tok(idx) + self.pos(positions)[None, :, :]

        # Pass through all Transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Final normalization and vocabulary projection.
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def moe_layers(self):
        # Return feed-forward modules containing a router.
        return [
            block.ffn
            for block in self.blocks
            if hasattr(block.ffn, "router")
        ]

