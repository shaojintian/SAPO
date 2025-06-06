Abstract
Reinforcement learning (RL) has become pivotal in advancing the reasoning capabilities of Large Language Models (LLMs), particularly for tasks requiring long chains of thought (CoT). However, existing methods struggle with two fundamental challenges: sparse rewards and long-term credit assignment. A single reward at the end of a thousand-token sequence provides a weak and diluted signal for optimizing early, critical reasoning steps. While recent value-based methods like VAPO have improved stability, they do not fully resolve this credit assignment problem. In this paper, we introduce SynAPO (Synergistic Actor-Critic Policy Optimization), a framework that builds upon a strong value-based baseline to create denser and more structured reward signals. We demonstrate through hypothetical experiments on the AIME 2024 benchmark that SynAPO significantly outperforms state-of-the-art methods, establishing a new paradigm for efficient and reliable RL in advanced reasoning tasks.
✨ Core Idea: From Sparse to Dense Rewards
Training LLMs for complex reasoning is like teaching a student to solve a multi-page math problem by only telling them if the final answer is right or wrong. The feedback is too sparse! The student doesn't know where they made a mistake.
SynAPO fixes this. It acts as a sophisticated tutor that provides dense, step-by-step feedback, transforming the single "right/wrong" signal into a rich tapestry of learning signals. It tells the model not just that it failed, but where it went wrong and how critical the error was.
🚀 Key Contributions
SynAPO integrates three novel, synergistic components to create this dense reward landscape:
🧠 Hierarchical Value Modeling (HVM): Decomposes a long reasoning chain into logical segments. It learns both the local, tactical correctness within a segment and the global, strategic value of the overall path, providing a structured scaffold for credit assignment.
🔍 Sub-path Contrastive Value Learning (SCVL): When the model generates multiple solutions, SCVL identifies the exact token where a bad solution diverges from a good one. It then applies a sharp, targeted contrastive loss, forcing the model to learn the value difference between a good and bad decision at the most critical point.
⚖️ Entropy-Aware Adaptive GAE (EA-GAE): Dynamically adjusts the bias-variance trade-off in advantage estimation. When the policy is uncertain (high entropy), it relies more on its stable (but biased) value estimate. When confident (low entropy), it trusts the high-variance but more accurate Monte Carlo returns. This leads to more stable and efficient training.
🔧 How It Works: A Deeper Dive
1. Hierarchical Value Modeling (HVM)
Instead of a single value function for the entire thousand-token sequence, HVM uses a two-tier system. This prevents the initial steps from being overshadowed by the final outcome.
V
local
V 
local
​
 
: Operates within segments (e.g., "Step 1", "Step 2"), judging short-term tactical quality.
V
global
V 
global
​
 
: Operates between segments, judging long-term strategic direction.
<p align="center">
<img src="https://i.imgur.com/pYcWn6t.png" alt="HVM Diagram" width="600px"/>
</p>
2. Sub-path Contrastive Value Learning (SCVL)
This is our "dense reward generator." By comparing a successful path (
T
+
T 
+
 
) with a failed one (
T
−
T 
−
 
) from the same starting point, we inject a powerful learning signal right where it matters most.
<p align="center">
<img src="https://i.imgur.com/83p144h.png" alt="SCVL Diagram" width="550px"/>
</p>
This contrastive loss, 
L
SCVL
L 
SCVL
 
, directly tells the value function: "The decision made here that led to path 
T
−
T 
−
 
 was suboptimal."
3. Entropy-Aware Adaptive GAE (EA-GAE)
We make the GAE's 
λ
p
λ 
p
​
 
 parameter adaptive not just to sequence length (like VAPO) but also to the policy's own uncertainty.
λ
p
(
l
,
H
t
)
=
1
−
1
α
⋅
l
⋅
(
1
+
β
⋅
H
t
(
s
t
)
)
λ 
p
​
 (l,H 
t
​
 )=1− 
α⋅l⋅(1+β⋅H 
t
​
 (s 
t
​
 ))
1
​
 
This allows the agent to be more cautious when exploring and more aggressive when it's confident, stabilizing the entire learning process.
📈 Hypothetical Results
Our hypothetical experiments show that SynAPO establishes a new state-of-the-art on the challenging AIME 2024 benchmark, significantly outperforming strong value-free (DAPO) and value-based (VAPO) baselines.
Model	AIME 2024 Acc. (%)
DAPO	50.0
VAPO (Baseline)	60.4
---	---
SynAPO (w/o HVM)	62.1
SynAPO (w/o SCVL)	63.5
SynAPO (w/o EA-GAE)	64.8
---	---
SynAPO (ours)	67.2
The training curves demonstrate SynAPO's superior sample efficiency and final performance, a direct result of its dense and structured reward signals.
<p align="center">
<img src="https://i.imgur.com/gK2YgLd.png" alt="Training Curve" width="650px"/>
</p>
🚀 Getting Started (Example Usage)
While this is a research project, here's a hypothetical example of how you might use SynAPO to fine-tune a model.
from transformers import AutoModelForCausalLM, AutoTokenizer
from synapo import SynAPOTrainer, SynAPOConfig

# 1. Load your base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

# 2. Define the SynAPO configuration
#    Easily enable or disable the key components.
config = SynAPOConfig(
    # Core VAPO/PPO params
    learning_rate=1e-5,
    ppo_epochs=4,
    # Enable SynAPO components
    hvm_enabled=True,
    scvl_enabled=True,
    ea_gae_enabled=True,
    # Component-specific params
    scvl_margin=0.5,
    scvl_weight=0.1,
    ea_gae_alpha=0.1,
    ea_gae_beta=0.05,
)

# 3. Initialize the trainer with your model and dataset
trainer = SynAPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    train_dataset=your_aime_dataset,
    eval_dataset=your_eval_dataset,
)

# 4. Launch training!
trainer.train()

print("Model fine-tuned with SynAPO's dense rewards!")
Use code with caution.
Python
📜 Citation
If you find our work useful, please consider citing:
@article{shao2025synapo,
  title={Dense Rewards, Deeper Reasoning: Synergistic Actor-Critic Policy Optimization for Long-Chain-of-Thought Tasks},
  author={Shao, Jintian and Cheng, Yiming and Shan, You and Zheng, Mingkai},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
Use code with caut
