# Train a Mixture-of-Experts LM and Watch It: Balance, Diagnostics, Upcycling and Serving

Train a small mixture-of-experts language model next to its dense twin and watch it the way the MoE training handbook (Hugging Face, 2026) says to watch a real one. Everything runs on a CPU in PyTorch, on a corpus you generate: eight domains, each a Markov chain with its own next-token rules, announced by a tag token and mixed in skewed proportions, so experts have something to specialise on and a router that specialises overloads a few of them. Because you wrote the generator, you also know the exact entropy floor, and every loss in the project is read as a gap above it. You build a two-layer decoder whose feed-forward module is pluggable, then the router piece by piece: softmax and sigmoid gating, top-k with a selection-only bias, renormalization, a scaling factor and a serving-time denominator; the router z-loss; the Switch-style load-balancing loss; and a capacity rule with two drop policies. These go into an MoE layer of eight experts with top-2 routing, matched to the dense model in active parameters (four times the stored feed-forward parameters), which records what it did on every forward pass. From those records you build the handbook's watch signals: MaxVio per layer, routing entropy, the top expert's weight, the routed output's size, the RMS logsumexp the z-loss tracks, the largest and smallest expert output norm over the median, the dropped share and, because the domains are planted, each layer's domain purity. Then the training choices: the fixed-step loss-free bias and quantile balancing, a training loop that runs every variant, upcycling a trained dense model into experts with exact parity at step 0, a run monitor with the handbook's alarms, a planted bug that makes one expert's optimizer update fifty times too large, and serving the top-2 model with one expert per token. What 24 seeds showed (300 steps of 16 sequences per model, 400 validation sequences, on the prod image). The MoE beats the dense model by 0.18 nats on every seed (0.47 against 0.65 above the floor). An aux loss with coefficient 0.1 cuts the worst layer's MaxVio from about 0.66 to 0.17 on every seed, with no measurable loss change (0.007 nats lower on average, lower on only 16 of 24 seeds); at coefficient 1.0 it costs 0.045 nats on every seed and drives the deeper layer's domain purity from 0.42 to 0.31, close to no specialisation. The loss-free bias with softmax scores matches the aux loss (0.006 nats lower on average, a margin within noise). Sigmoid scoring under the same bias costs 0.057 nats on every seed, with a top weight 0.16 lower: a flatter gate. Upcycled at the half-way point, the MoE starts with exactly the dense loss and with every token on the same two experts (MaxVio 3, the maximum), beats the dense run continued on the same batches by 0.047 nats on every seed, but loses by 0.135 to the MoE trained from scratch for the same number of steps: at this scale the from-scratch run has already caught up. At capacity factor 1, dropping by weight removes about 3 to 5% of assignments at no measurable cost, while dropping by position costs 0.051 nats on every seed. Served with one expert per token, dividing by the trained top-2 mass costs 0.071 nats against 0.086 for renormalizing over the survivor (better on 22 of 24 seeds, a small margin) and 0.175 with no renormalization. The planted bug fires the rogue-expert alarm on every seed, 20 to 40 steps in, while the loss alarm fires on one seed of 24 and the validation loss ends within noise of the clean run's; the healthy runs raise no alarm. The full run trains all seven models (plus two short continuations) for one seed in about 30 seconds and prints the dashboard. Its numbers come from a single seed and are noisy; the figures above say which comparisons held on every seed and which margins are small.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** domain_chains
- [x] **2.** sample_corpus
- [x] **3.** entropy_floor
- [x] **4.** DenseFFN
- [x] **5.** TinyLM
- [x] **6.** route_tokens
- [x] **7.** router_z_loss
- [x] **8.** load_balance_loss
- [x] **9.** apply_capacity
- [x] **10.** MoEFFN
- [x] **11.** param_counts
- [x] **12.** max_vio
- [x] **13.** router_sharpness
- [x] **14.** expert_norm_ratios
- [x] **15.** domain_purity
- [x] **16.** layer_dashboard
- [x] **17.** update_expert_bias
- [x] **18.** quantile_bias
- [x] **19.** evaluate_lm
- [x] **20.** train_lm
- [x] **21.** upcycle_model
- [x] **22.** RunMonitor
- [x] **23.** ExpertUpdateBug
- [x] **24.** set_serving
- [x] **25.** moe_study

## Results

```
1. Corpus and models
   8 domains with their own next-token rules, mix 0.75^d, tag token first; entropy floor 1.786 nats
   dense: 28,480 parameters, all active | MoE (8 experts, top-2): 78,144 total, 28,992 active

2. Validation loss after 300 steps (excess = loss minus the floor)
   dense FFN, width 128           2.4546  excess 0.6685
   MoE softmax, no balancing      2.2806  excess 0.4945
   MoE softmax, aux loss 0.1      2.2620  excess 0.4760
   MoE softmax, aux loss 1.0      2.2987  excess 0.5127
   MoE softmax, loss-free bias    2.2599  excess 0.4739
   MoE sigmoid, loss-free bias    2.2959  excess 0.5099
   Over 24 seeds: MoE beat dense on every seed (by 0.18 nats on average); aux 0.1 vs none and bias vs aux 0.1 were
   within noise; aux 1.0 cost 0.045 and sigmoid gating 0.057 on every seed.

3. The dashboard, per MoE layer (probe: 128 validation sequences)
   arm             layer  MaxVio  entropy  top_w  routed  rms_lse  norm_hi  norm_lo  purity
   none                0   0.577    1.655  0.688   0.771    2.513    1.056    0.880   0.298
   none                1   0.844    0.941  0.833   0.866    3.680    1.094    0.936   0.403
   aux                 0   0.096    1.690  0.680   0.766    2.506    1.061    0.926   0.284
   aux                 1   0.205    0.969  0.833   0.866    3.566    1.178    0.937   0.334
   aux_strong          0   0.081    1.853  0.640   0.746    2.328    1.033    0.952   0.271
   aux_strong          1   0.103    1.545  0.741   0.801    2.618    1.121    0.868   0.303
   bias                0   0.136    1.607  0.706   0.781    2.597    1.054    0.919   0.287
   bias                1   0.396    0.965  0.820   0.859    3.612    1.068    0.916   0.375
   sigmoid_bias        0   0.073    1.979  0.568   0.718    2.414    1.092    0.942   0.309
   sigmoid_bias        1   0.137    1.646  0.662   0.760    1.805    1.056    0.906   0.408
   Read it like this: balancing lowers MaxVio (every balancer did, on 22 to 24 of 24 seeds); the strong aux loss also
   lowers purity toward the no-specialisation level (about 0.28 plus sampling noise); sigmoid gating shows a lower
   top weight and a higher entropy than softmax, a flatter gate. Healthy expert norm ratios stay near 1.
   The monitor ran through the aux 0.1 run: alarms none

4. Upcycling the dense model at step 150
   dense loss at step 150 2.7833; upcycled MoE at step 0 differs by 9.5e-09
   MaxVio at the start: [3.0, 3.0] (every token on the same two experts); after 150 steps: [0.159, 0.181]
   after 300 steps in total: dense 2.4546 | upcycled 2.4008 | MoE from scratch 2.2620
   Over 24 seeds the upcycled model beat the continued dense run every time (by 0.047) and lost to the MoE trained
   from scratch every time (by 0.135): here the from-scratch run catches up within the same budget.

5. Capacity factor on the trained aux 0.1 model (dropped share per layer, loss change)
   factor 1.25, drop by weight    dropped [0.0, 0.0]  loss -0.0000
   factor 1.0, drop by weight     dropped [0.0386, 0.0527]  loss +0.0015
   factor 1.0, drop by position   dropped [0.0386, 0.0537]  loss +0.0676
   factor 0.5, drop by weight     dropped [0.5, 0.5]  loss +0.1342
   Dropping by position cost about 0.05 nats more than dropping by weight at the same factor, on every seed.

6. Serving the top-2 model with one expert per token
   renormalized over the survivor     loss +0.0911
   divided by the trained top-2 mass  loss +0.0712
   no renormalization                 loss +0.1625
   Over 24 seeds the trained-mass denominator beat the survivor on 22 (by 0.015, a small margin); no renormalization
   was worst on every seed.

7. A planted failure: 80 more steps with expert 3 of layer 1 updating 50 times too fast
   clean loss 2.148 -> 2.064  val 2.1859  layer-1 norm_hi 1.15  first alarms none
   bug   loss 2.194 -> 2.069  val 2.1888  layer-1 norm_hi 5.63  first alarms {'rogue_expert@1': 20}
   The loss keeps falling and ends within noise of the clean run; the rogue-expert alarm fired on all 24 seeds.

Total 18 s
```
