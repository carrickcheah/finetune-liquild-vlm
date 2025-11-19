# Why massive models aren't always the answer: Scaling performance on a 3B VLM

In the race for AI, we often focus on 70B+ parameter models. But for real-world production (especially on edge devices), efficiency is king.

I spent the weekend experimenting with LiquidAI's 3B Vision Model to see just how "smart" a small model could get with the right fine-tuning strategy. I leveraged Modal for compute and LoRA for parameter-efficient fine-tuning.

---

## Technical Implementation & Constraints

I leveraged Modal's L40S GPUs and LoRA to fine-tune the model on CIFAR-100. Because it's open-source, I had full control over the training loop to maximize efficiency:

| Configuration | Value |
|--------------|-------|
| **Model Scale** | 3B parameters (packed into just 6GB) |
| **LoRA Config** | Reduced trainable parameters to 4.2M  |
| **Batch Strategy** | Batch size 4 with gradient accumulation of 4 = Effective batch size 16 |
| **Training Volume** | ~8,400 steps over 3 epochs |

---

## Results: Data Efficiency vs. Scale

![Project Banner](media/33.png)

### 1. The Efficiency Test (10% Data / 5K Images)

- **Baseline**: 54% accuracy (Zero-shot)
- **Result**: Improved to 62% (+8 points)
- **Insight**: Significant gains tuning just 0.14% of parameters 

![Project Banner](media/22.png) 

### 2. The Scale-Up (100% Data / 50K Images)

- **Convergence**: Loss dropped from ~3.45 to 0.0006
- **Training Accuracy**: 99.98% Mean Token Accuracy
- **Test Accuracy**: 86% on held-out data

![Project Banner](media/5.png) 

The jump from 5K to 50K samples delivered a massive **24% accuracy gain**, far exceeding expectations. It proves we can build high-performance, private AI solutions that run locally at a fraction of the cost of renting a closed model.

![Project Banner](media/44.png) 

---

## Future Work

To bridge the gap between 99% training accuracy and 86% test accuracy:

- **Augmentation**: Implementing aggressive image augmentations (MixUp/CutMix) to improve robustness
- **Optimization**: Tuning LoRA rank/alpha to squeeze out every bit of performance
- **Early Stopping**: Evaluate intermediate checkpoints to find optimal stopping point

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/carrickcheah/finetune-liquild-vlm

# Install dependencies
uv sync

# Run training
modal run src/ft_vlm/cli/train.py --config-file-name finetune_lfm_3B_full.yaml

# Run evaluation
modal run src/ft_vlm/cli/evaluate.py --config-file-name eval_lfm_3B_finetuned.yaml
```

---

## Tags

`#ArtificialIntelligence` `#OpenSource` `#LocalLLM` `#ComputerVision` `#TechInnovation` `#LiquidAI`
