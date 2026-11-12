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
