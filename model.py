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

