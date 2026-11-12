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

# Step 6 - route_tokens
def route_tokens(logits, k, score="softmax", renorm=True, scale=1.0, bias=None, denom_k=None):
    # Perform routing calculations in float32 for numerical stability.
    logits = logits.float()

    # Calculate routing scores.
    if score == "softmax":
        s = torch.softmax(logits, dim=-1)
    elif score == "sigmoid":
        s = torch.sigmoid(logits)
    else:
        raise ValueError(f"Unsupported routing score: {score}")

    # Apply the optional bias only for expert selection.
    # The original scores s remain unchanged.
    if bias is not None:
        bias = torch.as_tensor(
            bias,
            dtype=s.dtype,
            device=s.device
        )
        selection_scores = s + bias
    else:
        selection_scores = s

    # Select the top-k experts using the biased selection scores.
    idx = torch.topk(
        selection_scores,
        k=k,
        dim=-1
    ).indices

    # Gather the original, unbiased scores of the selected experts.
    w = s.gather(-1, idx)

    # Normalize the selected weights.
    if denom_k is not None:
        # Divide by the sum of the denom_k largest unbiased scores.
        denominator = torch.topk(
            s,
            k=denom_k,
            dim=-1
        ).values.sum(dim=-1, keepdim=True)

        w = w / denominator

    elif renorm:
        # Normalize using only the selected experts.
        w = w / w.sum(dim=-1, keepdim=True)

    # Apply the routing scaling factor.
    w = w * scale

    return idx, w, s

# Step 7 - router_z_loss
def router_z_loss(logits):
    # Convert logits to float32 for numerical stability.
    logits = logits.float()

    # Calculate log-sum-exp across the expert dimension.
    log_z = torch.logsumexp(logits, dim=-1)

    # Return the mean squared log-sum-exp over all tokens.
    return (log_z ** 2).mean()

# Step 8 - load_balance_loss
def load_balance_loss(scores, idx, n_experts):
    # Normalize each token's routing scores to sum to 1.
    # This also converts sigmoid scores into a probability distribution.
    scores = scores / scores.sum(dim=-1, keepdim=True)

    # Flatten expert assignments and count assignments per expert.
    counts = torch.bincount(
        idx.reshape(-1),
        minlength=n_experts
    )

    # Fraction of all assignments received by each expert.
    f = counts.to(dtype=scores.dtype) / idx.numel()

    # Mean normalized routing score for each expert.
    P = scores.mean(dim=0)

    # Switch-style load-balancing loss.
    return n_experts * (f * P).sum()

# Step 9 - apply_capacity
import math


def apply_capacity(idx, w, n_experts, capacity_factor, policy="probs"):
    # Number of tokens and experts selected per token.
    N, k = idx.shape

    # Calculate the maximum number of assignments each expert can accept.
    cap = math.ceil(capacity_factor * N * k / n_experts)

    # Flatten assignments in row-major order.
    flat_idx = idx.reshape(-1)

    # Determine the processing order.
    if policy == "probs":
        # Process assignments by descending routing weight.
        # Stable sorting preserves original order when weights are tied.
        order = torch.sort(
            -w.reshape(-1),
            stable=True
        ).indices

    elif policy == "position":
        # Process assignments in their original flattened order.
        order = torch.arange(
            flat_idx.numel(),
            device=idx.device
        )

    else:
        raise ValueError(f"Unsupported capacity policy: {policy}")

    # Get expert assignments in the selected processing order.
    ordered_idx = flat_idx[order]

    # Initially, all assignments are dropped.
    keep = torch.zeros(
        flat_idx.numel(),
        dtype=torch.bool,
        device=idx.device
    )

    # Each expert keeps only its first 'cap' assignments.
    for expert in range(n_experts):
        # Find positions belonging to the current expert.
        positions = torch.nonzero(
            ordered_idx == expert,
            as_tuple=True
        )[0]

        # Keep only the first 'cap' assignments.
        positions = positions[:cap]

        # Map the selected positions back to their original indices.
        keep[order[positions]] = True

    # Restore the original (N, k) shape.
    return keep.reshape(N, k)

# Step 10 - MoEFFN
class MoEFFN(nn.Module):
    def __init__(self, d, n_experts, hidden, k, score="softmax", renorm=True, scale=1.0):
        super().__init__()

        # Store routing configuration.
        self.n_experts = n_experts
        self.k = k
        self.score = score
        self.renorm = renorm
        self.scale = scale

        # Learned, bias-free router.
        self.router = nn.Linear(d, n_experts, bias=False)

        # Expert-specific weight matrices.
        self.w1 = nn.Parameter(
            torch.randn(n_experts, d, hidden) * d ** -0.5
        )

        self.w2 = nn.Parameter(
            torch.randn(n_experts, hidden, d) * hidden ** -0.5
        )

        # Selection-only bias buffer.
        self.register_buffer(
            "bias",
            torch.zeros(n_experts)
        )

        # Capacity and serving configuration.
        self.capacity_factor = None
        self.drop_policy = "probs"
        self.serve_k = None
        self.denom_k = None

        # Store diagnostics from the latest forward pass.
        self.stats = {}

    def forward(self, x):
        # Preserve the original input shape.
        original_shape = x.shape

        # Flatten all leading dimensions into the token dimension.
        h = x.reshape(-1, original_shape[-1])
        N = h.shape[0]

        # Calculate router logits.
        logits = self.router(h)

        # Select the routing k (training or serving).
        k = self.serve_k if self.serve_k is not None else self.k

        # Calculate expert indices, mixing weights and routing scores.
        idx, w, s = route_tokens(
            logits,
            k,
            self.score,
            self.renorm,
            self.scale,
            self.bias,
            self.denom_k
        )

        # Initially, all selected assignments are retained.
        kept = torch.ones_like(idx, dtype=torch.bool)

        # Apply the expert capacity constraint when configured.
        if self.capacity_factor is not None:
            kept = apply_capacity(
                idx,
                w,
                self.n_experts,
                self.capacity_factor,
                self.drop_policy
            )

            # Zero the weights of dropped assignments.
            w = w * kept.to(w.dtype)

        # Compute every expert's output for every token.
        expert_outputs = torch.stack(
            [
                F.gelu(h @ self.w1[e]) @ self.w2[e]
                for e in range(self.n_experts)
            ],
            dim=1
        )

        # Construct the dense expert mixing matrix.
        gate = torch.zeros(
            (N, self.n_experts),
            dtype=w.dtype,
            device=w.device
        )

        # Accumulate the selected expert weights.
        gate.scatter_add_(1, idx, w)

        # Combine expert outputs using the routing weights.
        y = (
            expert_outputs
            * gate.to(expert_outputs.dtype).unsqueeze(-1)
        ).sum(dim=1)

        # Restore the original input dimensions.
        y = y.reshape(original_shape)

        # Calculate router auxiliary losses.
        # These retain their computation graphs for training.
        self.aux = load_balance_loss(
            s,
            idx,
            self.n_experts
        )

        self.z = router_z_loss(logits)

        # Record diagnostics without retaining gradients.
        with torch.no_grad():

            # Number of retained assignments per expert.
            counts = torch.bincount(
                idx[kept].reshape(-1),
                minlength=self.n_experts
            )

            # Per-token log-sum-exp of router logits.
            lse = torch.logsumexp(
                logits.float(),
                dim=-1
            )

            # Mean L2 output norm of each expert.
            out_norm = torch.linalg.vector_norm(
                expert_outputs.detach().float(),
                ord=2,
                dim=-1
            ).mean(dim=0)

            # Store the exact diagnostic keys.
            self.stats = {
                "counts": counts.detach(),
                "idx": idx.detach(),
                "weights": w.detach(),
                "scores": s.detach(),
                "kept": kept.detach(),
                "lse": lse.detach(),
                "out_norm": out_norm.detach()
            }

        return y

# Step 11 - param_counts
def param_counts(model):
    # Count all model parameters. Buffers are excluded automatically.
    total = sum(
        p.numel() for p in model.parameters()
    )

    # Initially, assume every parameter is active.
    active = total

    # Subtract parameters belonging to experts not selected per token.
    for layer in model.moe_layers():
        # Number of experts not used by a token.
        inactive_experts = layer.n_experts - layer.k

        # Parameters belonging to one expert.
        expert_params = (
            layer.w1[0].numel()
            + layer.w2[0].numel()
        )

        # Remove inactive expert parameters from the active count.
        active -= inactive_experts * expert_params

    return {
        "total": int(total),
        "active": int(active)
    }

# Step 12 - max_vio
def max_vio(counts):
    # Convert counts to a float64 PyTorch tensor.
    counts = torch.as_tensor(counts, dtype=torch.float64)

    # Calculate the mean expert load.
    mean_load = counts.mean()

    # Avoid division by zero when no assignments are present.
    if mean_load.item() == 0:
        return 0.0

    # Calculate the maximum load violation.
    max_violation = counts.max() / mean_load - 1.0

    # Return a Python float.
    return float(max_violation.item())

# Step 13 - router_sharpness
def router_sharpness(scores, weights):
    # Convert scores and weights to float32 for numerical stability.
    scores = scores.float()
    weights = weights.float()

    # Normalize routing scores row-wise to obtain probability distributions.
    p = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Calculate entropy for each token.
    entropy_per_token = -(
        p * torch.log(p.clamp_min(1e-12))
    ).sum(dim=-1)

    # Mean routing entropy across all tokens.
    entropy = entropy_per_token.mean()

    # Find the largest mixing weight for each token.
    top_weights = weights.max(dim=-1).values

    # Mean largest mixing weight across tokens.
    top_weight = top_weights.mean()

    # Calculate the L2 norm of the final mixing weights per token.
    routed_size = torch.sqrt(
        (weights ** 2).sum(dim=-1)
    ).mean()

    # Collect the diagnostic metrics.
    result = {
        "entropy": float(entropy.detach().item()),
        "top_weight": float(top_weight.detach().item()),
        "routed_size": float(routed_size.detach().item())
    }

    # Calculate the largest-to-second-largest weight ratio when k > 1.
    if weights.shape[-1] > 1:
        top_two = torch.topk(
            weights,
            k=2,
            dim=-1
        ).values

        max1_over_max2 = (
            top_two[:, 0] /
            top_two[:, 1].clamp_min(1e-12)
        ).mean()

        result["max1_over_max2"] = float(
            max1_over_max2.detach().item()
        )

    return result

# Step 14 - expert_norm_ratios
def expert_norm_ratios(norms):
    # Convert the input into a float64 PyTorch tensor.
    norms = torch.as_tensor(norms, dtype=torch.float64)

    # Calculate the median using the 0.5 quantile.
    median = torch.quantile(norms, 0.5)

    # Calculate the largest and smallest norm relative to the median.
    max_ratio = norms.max() / median
    min_ratio = norms.min() / median

    # Return both ratios as Python floats.
    return (
        float(max_ratio.item()),
        float(min_ratio.item())
    )

# Step 15 - domain_purity
def domain_purity(idx, token_domains, n_experts, n_domains):
    # Ensure inputs are PyTorch tensors with integer indices.
    idx = torch.as_tensor(idx, dtype=torch.long)
    token_domains = torch.as_tensor(
        token_domains,
        dtype=torch.long,
        device=idx.device
    )

    # Get the number of tokens and selected experts per token.
    N, k = idx.shape

    # Repeat each token's domain for all its expert assignments.
    domains = token_domains.unsqueeze(1).expand(N, k)

    # Encode each (expert, domain) pair into a unique index.
    flat_indices = (
        idx.reshape(-1) * n_domains
        + domains.reshape(-1)
    )

    # Count assignments for every expert-domain combination.
    counts = torch.bincount(
        flat_indices,
        minlength=n_experts * n_domains
    ).reshape(n_experts, n_domains)

    # Calculate the total assignments received by each expert.
    expert_totals = counts.sum(dim=1)

    # Consider only experts that received at least one assignment.
    used = expert_totals > 0

    # Return zero when no expert received any assignments.
    if not used.any():
        return 0.0

    # Calculate the largest domain share for each used expert.
    purities = (
        counts.max(dim=1).values[used].float()
        / expert_totals[used].float()
    )

    # Return the mean domain purity as a Python float.
    return float(purities.mean().item())

# Step 16 - layer_dashboard
def layer_dashboard(model, x, domains=None, n_domains=8):
    # Remember the model's original training/evaluation mode.
    was_training = model.training

    try:
        # Switch to evaluation mode.
        model.eval()

        # Perform exactly one forward pass without computing gradients.
        with torch.no_grad():
            model(x)

            # Collect diagnostics from every MoE layer.
            rows = []

            for layer in model.moe_layers():
                stats = layer.stats

                # 1. Maximum load violation.
                maxvio = max_vio(stats["counts"])

                # 2. Router sharpness metrics.
                sharpness = router_sharpness(
                    stats["scores"],
                    stats["weights"]
                )

                # 3. RMS log-sum-exp.
                rms_lse = torch.sqrt(
                    torch.mean(stats["lse"].float() ** 2)
                ).item()

                # 4. Expert output norm ratios.
                norm_hi, norm_lo = expert_norm_ratios(
                    stats["out_norm"]
                )

                # 5. Fraction of dropped assignments.
                dropped = 1.0 - stats["kept"].float().mean().item()

                # Build the diagnostic dictionary.
                row = {
                    "maxvio": round(maxvio, 3),
                    "entropy": round(sharpness["entropy"], 3),
                    "top_weight": round(sharpness["top_weight"], 3),
                    "routed_size": round(sharpness["routed_size"], 3),
                    "rms_lse": round(rms_lse, 3),
                    "norm_hi": round(norm_hi, 3),
                    "norm_lo": round(norm_lo, 3),
                    "dropped": round(dropped, 4)
                }

                # 6. Calculate domain purity when domain labels are provided.
                if domains is not None:
                    # Expand each sequence's domain across its token positions.
                    token_domains = (
                        torch.as_tensor(domains)
                        .reshape(-1, 1)
                        .expand(-1, x.shape[1])
                        .reshape(-1)
                    )

                    purity = domain_purity(
                        stats["idx"],
                        token_domains,
                        layer.n_experts,
                        n_domains
                    )

                    row["purity"] = round(purity, 3)

                rows.append(row)

        return rows

    finally:
        # Restore the model's original mode, even if an error occurs.
        model.train(was_training)

# Step 17 - update_expert_bias
def update_expert_bias(bias, counts, rate, rule="sign"):
    # Convert expert counts to float64 for accurate calculations.
    counts = torch.as_tensor(
        counts,
        dtype=torch.float64,
        device=bias.device
    )

    # Calculate the mean expert load and the load error.
    mean_load = counts.mean()
    error = mean_load - counts

    # Calculate the balancing step according to the selected rule.
    if rule == "sign":
        step = torch.sign(error)

    elif rule == "centered":
        signs = torch.sign(error)
        step = signs - signs.mean()

    elif rule == "linear":
        # Avoid division by zero when all experts have zero load.
        if mean_load.item() == 0:
            step = torch.zeros_like(error)
        else:
            step = error / mean_load

    else:
        raise ValueError(f"Unsupported balancing rule: {rule}")

    # Apply the update and preserve the original bias dtype.
    updated_bias = bias.to(torch.float64) + rate * step

    return updated_bias.to(dtype=bias.dtype)

