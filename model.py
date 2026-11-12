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

