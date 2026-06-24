# Optimisation Changes

## SwiGLU MLP
Replaced the standard GELU feed-forward block with SwiGLU (Swish-Gated Linear Unit), as used in LLaMA and PaLM. Instead of a single linear projection followed by an activation, SwiGLU uses two parallel projections — one passed through a Swish (SiLU) activation acts as a gate that controls how much of the second projection flows through. This gating mechanism gives the network a more expressive non-linearity per parameter, consistently improving perplexity over GELU in modern language models. The hidden dimension is set to `8/3 × d_model` to keep parameter count comparable to the 4× GELU expansion.

The scaled residual initialisation target was updated from `net.2.weight` to `down.weight` to match the new projection naming.

## AdamW beta2 = 0.95
Changed the second Adam momentum coefficient from the PyTorch default of 0.999 to 0.95, following the GPT-2 and Chinchilla training setups. A high beta2 accumulates gradient history slowly, which can cause the optimiser to hold onto stale signal in later training stages. Lowering it to 0.95 makes the optimiser more responsive to recent gradients, reducing the oscillation seen in the final 300 steps of the previous run where the loss plateaued at ~1.270 instead of descending.

## Cosine LR floor lowered (eta_min)
Reduced the cosine annealing floor from `lr × 0.05` to `lr × 0.01`. The previous floor meant the learning rate never decayed below ~2×10⁻⁵, which was too high for fine-grained descent in the final epochs. Lowering it to ~4×10⁻⁶ gives the scheduler more room to make precise weight updates at the end of training, directly targeting the plateau observed after step 660.
