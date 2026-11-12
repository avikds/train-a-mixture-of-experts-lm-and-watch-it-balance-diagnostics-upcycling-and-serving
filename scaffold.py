"""
Train a Mixture-of-Experts LM and Watch It: Balance, Diagnostics, Upcycling and Serving scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Train a mixture-of-experts LM next to its dense twin and watch it, in about 30 seconds on one CPU.

This run is one seed of the study the project description reports over 24 seeds. Its numbers are noisy: gaps of a few
hundredths of a nat can shrink, grow or flip on another seed. The printed notes say which comparisons held on every seed."""
from solution import *
import time
import torch

torch.set_num_threads(1)
T0 = time.time()
SEED = 0
R = moe_study(seed=SEED, steps=300)
A = R["arms"]
floor = R["floor"]

print("1. Corpus and models")
print(f"   8 domains with their own next-token rules, mix 0.75^d, tag token first; entropy floor {floor:.3f} nats")
pd, pm = A["dense"]["params"], A["aux"]["params"]
print(f"   dense: {pd['total']:,} parameters, all active | MoE (8 experts, top-2): {pm['total']:,} total, {pm['active']:,} active")

print("\n2. Validation loss after 300 steps (excess = loss minus the floor)")
names = {"dense": "dense FFN, width 128", "none": "MoE softmax, no balancing", "aux": "MoE softmax, aux loss 0.1",
         "aux_strong": "MoE softmax, aux loss 1.0", "bias": "MoE softmax, loss-free bias", "sigmoid_bias": "MoE sigmoid, loss-free bias"}
for k, label in names.items():
    print(f"   {label:30s} {A[k]['val']:.4f}  excess {A[k]['val'] - floor:.4f}")
print("   Over 24 seeds: MoE beat dense on every seed (by 0.18 nats on average); aux 0.1 vs none and bias vs aux 0.1 were")
print("   within noise; aux 1.0 cost 0.045 and sigmoid gating 0.057 on every seed.")

print("\n3. The dashboard, per MoE layer (probe: 128 validation sequences)")
print("   arm             layer  MaxVio  entropy  top_w  routed  rms_lse  norm_hi  norm_lo  purity")
for k in ("none", "aux", "aux_strong", "bias", "sigmoid_bias"):
    for i, r in enumerate(A[k]["dash"]):
        print(f"   {k:15s} {i:5d}  {r['maxvio']:6.3f}  {r['entropy']:7.3f}  {r['top_weight']:5.3f}  {r['routed_size']:6.3f}  "
              f"{r['rms_lse']:7.3f}  {r['norm_hi']:7.3f}  {r['norm_lo']:7.3f}  {r['purity']:6.3f}")
print("   Read it like this: balancing lowers MaxVio (every balancer did, on 22 to 24 of 24 seeds); the strong aux loss also")
print("   lowers purity toward the no-specialisation level (about 0.28 plus sampling noise); sigmoid gating shows a lower")
print("   top weight and a higher entropy than softmax, a flatter gate. Healthy expert norm ratios stay near 1.")
print(f"   The monitor ran through the aux 0.1 run: alarms {R['healthy_alarms'] or 'none'}")

u = R["upcycle"]
print("\n4. Upcycling the dense model at step 150")
print(f"   dense loss at step 150 {R['dense_half']:.4f}; upcycled MoE at step 0 differs by {u['parity_gap']:.1e}")
print(f"   MaxVio at the start: {[r['maxvio'] for r in u['dash_start']]} (every token on the same two experts); after 150 steps: "
      f"{[r['maxvio'] for r in u['dash_end']]}")
print(f"   after 300 steps in total: dense {A['dense']['val']:.4f} | upcycled {u['val']:.4f} | MoE from scratch {A['aux']['val']:.4f}")
print("   Over 24 seeds the upcycled model beat the continued dense run every time (by 0.047) and lost to the MoE trained")
print("   from scratch every time (by 0.135): here the from-scratch run catches up within the same budget.")

print("\n5. Capacity factor on the trained aux 0.1 model (dropped share per layer, loss change)")
for k, label in (("cf1.25", "factor 1.25, drop by weight"), ("cf1.0", "factor 1.0, drop by weight"),
                 ("cf1.0_position", "factor 1.0, drop by position"), ("cf0.5", "factor 0.5, drop by weight")):
    c = R["capacity"][k]
    print(f"   {label:30s} dropped {c['dropped']}  loss {c['delta']:+.4f}")
print("   Dropping by position cost about 0.05 nats more than dropping by weight at the same factor, on every seed.")

print("\n6. Serving the top-2 model with one expert per token")
for dk, label in ((1, "renormalized over the survivor"), (2, "divided by the trained top-2 mass"), (8, "no renormalization")):
    print(f"   {label:34s} loss {R['serving'][f'k1_over_top{dk}']:+.4f}")
print("   Over 24 seeds the trained-mass denominator beat the survivor on 22 (by 0.015, a small margin); no renormalization")
print("   was worst on every seed.")

f = R["failure"]
print("\n7. A planted failure: 80 more steps with expert 3 of layer 1 updating 50 times too fast")
for k in ("clean", "bug"):
    r = f[k]
    print(f"   {k:5s} loss {r['loss_first10']:.3f} -> {r['loss_last10']:.3f}  val {r['val']:.4f}  layer-1 norm_hi {r['norm_hi']:.2f}  "
          f"first alarms {r['first'] or 'none'}")
print("   The loss keeps falling and ends within noise of the clean run; the rogue-expert alarm fired on all 24 seeds.")
print(f"\nTotal {time.time() - T0:.0f} s")

