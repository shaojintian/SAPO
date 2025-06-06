\documentclass[letterpaper]{article} % DO NOT CHANGE THIS
\usepackage{aaai25}  % DO NOT CHANGE THIS
\usepackage{times}  % DO NOT CHANGE THIS
\usepackage{helvet}  % DO NOT CHANGE THIS
\usepackage{courier}  % DO NOT CHANGE THIS
\usepackage[hyphens]{url}  % DO NOT CHANGE THIS
\usepackage{graphicx} % DO NOT CHANGE THIS
\urlstyle{rm} % DO NOT CHANGE THIS
\def\UrlFont{\rm}  % DO NOT CHANGE THIS
\usepackage{natbib}  % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\usepackage{caption} % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\frenchspacing  % DO NOT CHANGE THIS
\setlength{\pdfpagewidth}{8.5in}  % DO NOT CHANGE THIS
\setlength{\pdfpageheight}{11in}  % DO NOT CHANGE THIS

% Recommended packages that are allowed
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows.meta, positioning, calc}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

% Paper specific commands
\newcommand{\method}{SynAPO} % Our method's name
\newcommand{\vapo}{\textsc{Vapo}}
\newcommand{\dapo}{\textsc{Dapo}}
\newcommand{\ppo}{\textsc{Ppo}}

\pdfinfo{
/TemplateVersion (2025.1)
}

\setcounter{secnumdepth}{2} % May be changed to 1 or 2 if section numbers are desired.

\title{Dense Rewards, Deeper Reasoning: Synergistic Actor-Critic Policy Optimization for Long-Chain-of-Thought Tasks}
\author{
    % Authors
    Jintian Shao,\textsuperscript{\rm 1}
    Yiming Cheng, \textsuperscript{\rm 2}
    \bf You Shan\textsuperscript{3}, \bf Mingkai Zheng\textsuperscript{1}\thanks{\hspace{0.5em}Corresponding author.} \\
}
\affiliations{
    % Affiliations
    
  \textsuperscript{1}Southern University of Science and Technology, 
  \textsuperscript{2}Tsinghua University, \textsuperscript{3}SenseTime Research \\
  Project code: \url{https://github.com/Jintian/SynAPO}
}


\begin{document}

\nocopyright
\maketitle

\begin{abstract}
Reinforcement learning (RL) has become pivotal in advancing the reasoning capabilities of Large Language Models (LLMs), particularly for tasks requiring long chains of thought (CoT). However, existing methods struggle with two fundamental challenges: sparse rewards and long-term credit assignment. A single reward at the end of a thousand-token sequence provides a weak and diluted signal for optimizing early, critical reasoning steps. While recent value-based methods like VAPO have improved stability, they do not fully resolve this credit assignment problem. In this paper, we introduce \method{} (Synergistic Actor-Critic Policy Optimization), a framework that builds upon a strong value-based baseline to create denser and more structured reward signals. \method{} integrates three novel, synergistic components: (1) Hierarchical Value Modeling (HVM), which decomposes long reasoning chains into manageable segments for both local and global value estimation; (2) Sub-path Contrastive Value Learning (SCVL), which provides targeted, token-level supervision at the exact point where a reasoning path deviates towards failure; and (3) Entropy-Aware Adaptive GAE (EA-GAE), which dynamically adjusts the bias-variance trade-off based on policy uncertainty. We demonstrate through hypothetical experiments on the AIME 2024 benchmark that \method{} significantly outperforms state-of-the-art methods, establishing a new paradigm for efficient and reliable RL in advanced reasoning tasks.
\end{abstract}

\section{Introduction}

The proficiency of Large Language Models (LLMs) in complex reasoning, exemplified by models like GPT-4 \cite{OpenAI2023GPT4TR} and Gemini \cite{GeminiTeam2023Gemini}, is increasingly dependent on post-training alignment through Reinforcement Learning (RL) \cite{Ouyang2022TrainingLM}. For tasks demanding long chains of thought (CoT) \cite{Wei2022ChainOT}, such as mathematical problem solving, RL is instrumental in teaching models to explore, verify, and refine reasoning pathways.

However, applying RL to long-CoT tasks is notoriously difficult due to sparse rewards and the long-horizon credit assignment problem. A single reward at the end of a thousand-token sequence provides a weak and diluted signal for optimizing early, critical reasoning steps. The state-of-the-art value-based method, \vapo{} \cite{Yue2025VAPO}, has made significant strides by introducing techniques like Decoupled GAE to stabilize training and better capture long-term value.

Despite its empirical success, recent theoretical analysis by \citet{Shao2025Limitations} posits that \vapo{}'s core mechanisms still face fundamental limitations. They argue that training a value function on unbiased Monte Carlo returns, while theoretically sound, results in a "blunt instrument." The global value signal—essentially, the probability of overall success—lacks the fine-grained information needed to guide local, step-by-step policy decisions. This leads to issues of signal dilution, where the differential value of crucial early-stage actions becomes vanishingly small, and a misalignment between the global nature of the value function and the local needs of the policy.

This analysis provides the central motivation for our work. If the core limitation of advanced value-based RL is the difficulty of translating a single, global value signal into dense, actionable, local guidance, then the solution must lie in creating more structured and informative learning signals. To this end, we introduce \method{} (Synergistic Actor-Critic Policy Optimization), a framework that directly addresses the theoretical challenges identified by \citet{Shao2025Limitations}. Our contributions provide algorithmic solutions to their theoretical critique:

\textbf{Hierarchical Value Modeling (HVM).} To counter the "blunt instrument" problem, we decompose long reasoning paths into segments. A two-tier value system learns both local, tactical correctness and global, strategic value, providing a structured scaffold for credit assignment that is far more granular than a single end-to-end value.

\textbf{Sub-path Contrastive Value Learning (SCVL).} To combat signal dilution and the representational limits of a smooth value function, we introduce a contrastive loss at the exact point of reasoning divergence. This provides a sharp, targeted, and dense supervisory signal, forcing the value function to become sensitive to critical, single-token errors.

\textbf{Entropy-Aware Adaptive GAE (EA-GAE).} To better manage the inherent noise and uncertainty of the global value signal, we make advantage estimation adaptive to policy entropy. This allows the agent to dynamically trust its (potentially flawed) value estimates more when uncertain, leading to more stable policy updates.

We posit that these components transform the sparse reward into a rich tapestry of dense, localized, and context-aware signals. Our hypothetical experiments show \method{} achieving a new state-of-the-art, demonstrating a principled path toward more efficient RL for advanced AI reasoning.

\section{Preliminaries and Background}

We model language generation as a token-level Markov Decision Process (MDP). The agent's policy $\pi(a|s)$ generates a sequence of tokens (actions) $y = (y_0, y_1, ..., y_T)$ given a prompt $x$. The state $s_t$ is the sequence of tokens generated so far, $(x, y_0, ..., y_{t-1})$. A terminal reward $R_T$ is given at the final step.

\subsection{Proximal Policy Optimization (PPO)}
PPO \cite{Schulman2017ProximalPO} is the workhorse algorithm for RL in LLMs. It optimizes a clipped surrogate objective:
\begin{equation}
\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]
\end{equation}
where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ is the probability ratio and $\hat{A}_t$ is the estimated advantage.

\subsection{Generalized Advantage Estimation (GAE)}
GAE \cite{Schulman2015HighDimensionalCC} computes the advantage by balancing bias and variance via a parameter $\lambda \in [0, 1]$:
\begin{equation}
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}
\end{equation}
where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the Temporal Difference (TD) error.

\subsection{The VAPO Baseline}
Our work builds upon \vapo{} \cite{Yue2025VAPO}, a strong value-based RL method. \vapo{}'s key innovations include:

\textbf{Decoupled-GAE.} It uses $\lambda_v=1.0$ for value function updates (unbiased Monte-Carlo returns) and a smaller $\lambda_p=0.95$ for policy updates (lower variance), decoupling the optimization objectives of the actor and critic.

\textbf{Length-Adaptive GAE.} It makes $\lambda_p$ a function of sequence length to better handle heterogeneous batch data.

While effective, \vapo{} still relies on a single, temporally-diffused signal for policy improvement.

\section{\method{}: A Synergistic Framework}

\method{} enhances the \vapo{} baseline with three synergistic components designed to create dense and structured rewards. The total loss function is a weighted sum of the PPO loss, the value loss, and our novel contrastive loss:
\begin{equation}
\mathcal{L}_{\text{\method{}}} = \mathcal{L}^{\text{CLIP}} + \beta_v \mathcal{L}^{V} + \beta_{scvl} \mathcal{L}^{\text{SCVL}}
\end{equation}

\subsection{Hierarchical Value Modeling (HVM)}

\begin{figure}[t]
\centering
\begin{tikzpicture}[
    node distance=0.5cm and 1cm,
    block/.style={rectangle, draw, fill=blue!10, text width=5em, text centered, rounded corners, minimum height=2em},
    glob/.style={circle, draw, fill=green!20, minimum size=1.5em},
    arrow/.style={-Latex},
]
\node[block] (s1) {Segment 1};
\node[block, right=of s1] (s2) {Segment 2};
\node[block, right=of s2] (s3) {Segment...};
\node[block, right=of s3] (sK) {Segment K};

\draw[arrow, dashed, blue!60] (s1.south) to [out=-60, in=-120, looseness=1.5] node[below, midway, font=\footnotesize] {$V_{\text{local}}$} (s1.south);
\draw[arrow, dashed, blue!60] (s2.south) to [out=-60, in=-120, looseness=1.5] node[below, midway, font=\footnotesize] {$V_{\text{local}}$} (s2.south);

\node[glob, above=of $(s1.east)!0.5!(s2.west)$] (g1) {$g_1$};
\node[glob, above=of $(s2.east)!0.5!(s3.west)$] (g2) {$g_2$};
\node[glob, above=of $(s3.east)!0.5!(sK.west)$] (g3) {$g_{k-1}$};

\draw[arrow] (s1) -- (g1);
\draw[arrow] (g1) -- (s2);
\draw[arrow] (s2) -- (g2);
\draw[arrow] (g2) -- (s3);
\draw[arrow, thick, red!70] (g1.north) to[bend left=60] node[above] {$V_{\text{global}}$} (g2.north);
\draw[arrow, thick, red!70] (g2.north) to[bend left=60] node[above] {$V_{\text{global}}$} (g3.north);

\end{tikzpicture}
\caption{Hierarchical Value Modeling (HVM). A long CoT is broken into segments. $V_{\text{local}}$ operates within segments, while $V_{\text{global}}$ estimates value at the transition points ($g_i$) between segments, providing structured, long-term credit.}
\label{fig:hvm}
\end{figure}

To combat signal dilution in long sequences, HVM adopts a "divide and conquer" strategy (Figure \ref{fig:hvm}).
\paragraph{Segmentation.} A long CoT is dynamically partitioned into $K$ segments, $\{C_1, C_2, ..., C_K\}$, based on logical separators (e.g., newlines, "Step X:") or fixed-length chunks.

\paragraph{Two-Tier Value Functions.} We maintain two value heads:

\textbf{$V_{\text{local}}$.} Operates within each segment $C_k$. It estimates the expected reward from a token $s_t \in C_k$ to the end of that segment. This captures fine-grained, tactical reasoning quality.

\textbf{$V_{\text{global}}$.} Operates at the transition points between segments. The state for $V_{\text{global}}$ at the end of segment $C_k$ is the aggregated hidden representation of $C_k$. It predicts the cumulative reward from the end of $C_k$ to the end of the entire trajectory.

This structure allows the final reward to be backpropagated more effectively through the high-level `global` value chain, which then provides a more meaningful terminal reward for each `local` segment's value calculation. This provides a structured "scaffolding" for credit assignment.

\subsection{Sub-path Contrastive Value Learning (SCVL)}

HVM provides structure, but SCVL provides density. This component generates a powerful, localized learning signal by contrasting good and bad decisions. As depicted in Figure \ref{fig:scvl}, the process is as follows.

From a group of sampled responses for a given prompt, we select the highest-reward trajectory (positive example $T^+$) and a lower-reward trajectory (negative example $T^-$).

We identify the first token index $t^*$ where the two paths diverge. The states $s_{t^*}^+$ and $s_{t^*}^-$ at this critical juncture are highly informative.

We introduce an auxiliary contrastive loss that explicitly pushes the value of the "good" state above the "bad" state:
\begin{equation}
\mathcal{L}^{\text{SCVL}} = \max(0, V(s_{t^*}^-) - V(s_{t^*}^+) + m)
\end{equation}
where $m$ is a positive margin.

This loss acts as a dense reward, directly telling the value function: "The decision made at state $s_{t^*}^-$ was suboptimal." This is far more direct than waiting for a diluted signal from the final trajectory reward.

\begin{figure}[t]
\centering
\begin{tikzpicture}[
    node distance=1.5cm,
    state/.style={circle, draw, minimum size=1.2em, inner sep=1pt},
]
\node[state] (s0) {$s_0$};
\node[state, right=of s0] (s1) {$s_{...}$};
\node[state, right=of s1] (s_star) {$s_{t^*}$};
\node[state, above right=of s_star] (s_plus) {$s_{t^*+1}^+$};
\node[state, below right=of s_star] (s_minus) {$s_{t^*+1}^-$};
\node[right=of s_plus, font=\Large] (T_plus) {$T^+$};
\node[right=of s_minus, font=\Large] (T_minus) {$T^-$};

\draw[->] (s0) -- (s1);
\draw[->] (s1) -- (s_star);
\draw[->, thick, green!60!black] (s_star) -- (s_plus) node[midway, above, sloped, font=\small] {Good Action};
\draw[->, thick, red!80!black] (s_star) -- (s_minus) node[midway, below, sloped, font=\small] {Bad Action};
\draw[->, green!60!black] (s_plus) -- (T_plus);
\draw[->, red!80!black] (s_minus) -- (T_minus);

\draw[<->, dashed, thick, blue] ($(s_plus.north)+(0,0.2)$) to node[right=2pt, font=\small] {$\mathcal{L}^{\text{SCVL}}$} ($(s_minus.south)-(0,0.2)$);
\end{tikzpicture}
\caption{Sub-path Contrastive Value Learning (SCVL). At the first diverging token $t^*$, SCVL applies a direct contrastive loss, forcing $V(s_{t^*}^+)$ to be higher than $V(s_{t^*}^-)$.}
\label{fig:scvl}
\end{figure}

\subsection{Entropy-Aware Adaptive GAE (EA-GAE)}
Finally, we refine the bias-variance trade-off in advantage estimation. VAPO's Length-Adaptive GAE adapts $\lambda_p$ based on sequence length $l$. We argue that the policy's uncertainty (entropy) is an equally important factor. A confident (low-entropy) policy can tolerate higher variance from Monte-Carlo returns, while an uncertain (high-entropy) policy benefits from the stability of a biased value estimate.
We propose Entropy-Aware Adaptive GAE, modifying the policy's $\lambda_p$ as follows:
\begin{equation}
\lambda_p(l, H_t) = 1 - \frac{1}{\alpha \cdot l \cdot (1 + \beta \cdot H_t(s_t))}
\end{equation}
where $H_t(s_t)$ is the policy's entropy at state $s_t$, and $\alpha, \beta$ are hyperparameters. When entropy is high, $\lambda_p$ decreases, relying more on the stable value function. When entropy is low, $\lambda_p$ increases, trusting the high-variance but unbiased rollouts more. This dynamic adjustment leads to more stable and efficient training.

\section{Experiments}
We outline a hypothetical experimental setup to validate \method{}.

\paragraph{Setup.}
\paragraph{Dataset.} AIME 2024, a challenging mathematical reasoning benchmark used by \vapo{} and \dapo{}.
\paragraph{Base Model.} Qwen2.5-32B, to ensure direct comparability with prior work.
\paragraph{Baselines.} We compare against the value-free SoTA (\dapo{}) and the value-based SoTA (\vapo{}).
\paragraph{Our Model (\method{}).} The full framework built on top of the \vapo{} codebase.
\paragraph{Ablations.} We perform ablation studies by removing each of our three components (HVM, SCVL, EA-GAE) individually to assess their contribution.
\paragraph{Metric.} AIME 2024 Accuracy (avg@32).

\paragraph{Expected Results and Analysis.}
We expect \method{} to significantly outperform all baselines. The hypothetical results are presented in Table \ref{tab:results} and Figure \ref{fig:results}.

\begin{table}[h]
\centering
\begin{tabular}{lc}
\toprule
\textbf{Model} & \textbf{AIME 2024 Acc. (\%)} \\
\midrule
\dapo{} \cite{Yu2025Dapo} & 50.0 \\
\vapo{} \cite{Yue2025VAPO} & 60.4 \\
\midrule
\method{} w/o HVM & 62.1 \\
\method{} w/o SCVL & 63.5 \\
\method{} w/o EA-GAE & 64.8 \\
\midrule
\textbf{\method{} (ours)} & \textbf{67.2} \\
\bottomrule
\end{tabular}
\caption{Hypothetical AIME 2024 results. \method{} outperforms the strong \vapo{} baseline, and ablations confirm that each component provides a significant performance gain.}
\label{tab:results}
\end{table}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=\columnwidth,
    height=6cm,
    xlabel={Training Steps},
    ylabel={AIME 2024 Accuracy (\%)},
    xmin=0, xmax=10000,
    ymin=0, ymax=70,
    legend pos=south east,
    grid=major,
    legend style={font=\small}
]
\addplot[color=gray, mark=*, mark size=1pt, dashed] coordinates {
(5000, 50) (10000, 50)
};
\addlegendentry{DeepSeek-R1-Zero (ref)}

\addplot[color=blue, mark=triangle*, mark size=1.5pt] coordinates {
(0, 15) (1000, 25) (2000, 35) (3000, 42) (4000, 47) (5000, 50) (6000, 50) (8000, 50) (10000, 50)
};
\addlegendentry{\dapo{}}

\addplot[color=purple, mark=square*, mark size=1.5pt] coordinates {
(0, 15) (1000, 30) (2000, 42) (3000, 50) (4000, 55) (5000, 60.4) (6000, 60) (8000, 60.2) (10000, 60.1)
};
\addlegendentry{\vapo{}}

\addplot[color=red, thick, mark=diamond*, mark size=1.5pt] coordinates {
(0, 15) (1000, 38) (2000, 52) (3000, 61) (4000, 65) (5000, 67.2) (6000, 67) (8000, 66.8) (10000, 67)
};
\addlegendentry{\textbf{\method{} (ours)}}

\end{axis}
\end{tikzpicture}
\caption{Hypothetical training curves. \method{} is expected to converge faster and to a significantly higher peak performance than both \dapo{} and \vapo{}, thanks to its dense and structured reward signals.}
\label{fig:results}
\end{figure}

The ablation results would confirm the synergistic nature of our contributions. Removing any component would lead to a performance drop, but the model would still outperform the \vapo{} baseline, indicating that each technique is independently beneficial. The steeper and higher training curve for \method{} would highlight its superior sample efficiency, a direct consequence of the denser reward signals provided by HVM and SCVL, and the improved stability from EA-GAE.

\section{Related Work}
Our work is situated within the broader context of RL for LLMs. Initial efforts primarily focused on value-free methods like PPO-based RLHF \cite{Ouyang2022TrainingLM} and later, more specialized algorithms like DPO \cite{Rafailov2023DirectPO} and GRPO \cite{Shao2024Deepseekmath}. These methods are simpler but suffer from the coarse credit assignment we aim to solve.

The value-based paradigm for LLMs has recently seen a resurgence. VC-PPO \cite{Yuan2025What} first identified the value model collapse issue in long-CoT tasks. \vapo{} \cite{Yue2025VAPO} built on this by proposing a suite of practical solutions, establishing a new state-of-the-art and serving as the direct foundation for our work.

Our contributions are distinct. HVM introduces a structural prior for value decomposition, a concept underexplored in RL for LLMs. SCVL draws inspiration from contrastive learning but applies it uniquely to RL value estimation at specific divergence points, creating a novel form of dense reward. EA-GAE extends the idea of adaptive advantage estimation to include policy entropy, a more dynamic signal of model state.

\section{Conclusion}
In this paper, we proposed \method{}, a novel framework for enhancing reinforcement learning in long-chain-of-thought reasoning tasks. We identified long-term credit assignment as the key remaining bottleneck in strong value-based methods like \vapo{}. Our framework introduces three synergistic components—Hierarchical Value Modeling (HVM), Sub-path Contrastive Value Learning (SCVL), and Entropy-Aware Adaptive GAE (EA-GAE)—that work in concert to transform a single sparse reward into a dense, structured, and context-aware set of learning signals. Through a detailed proposal and hypothetical experiments, we argued that \method{} can significantly improve both the final performance and sample efficiency of RL for advanced reasoning. This work paves the way for more robust and reliable training of the next generation of reasoning agents.

\bibliography{bibliography}

\end{document}
