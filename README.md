# OCR-Free Key Information Extraction using Vision-Language Models

> **Summer Research Internship, Indian Institute of Technology Ropar**
> **Supervisor:** Dr. Gyanendra Singh

---

## 1. Overview

Key Information Extraction (KIE) requires identifying semantically relevant entities $e \in \mathcal{E}$ from a visually rich document $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ by jointly reasoning over text content, spatial layout $\{(x_i, y_i, w_i, h_i)\}$, and visual context. Classical KIE factorizes this as a pipeline of independently-optimized modules:

$$
p(\mathcal{E} \mid \mathbf{I}) = p_{\text{NER}}\big(\mathcal{E} \mid \hat{T}\big) \cdot p_{\text{Recog}}\big(\hat{T} \mid \hat{R}\big) \cdot p_{\text{Det}}\big(\hat{R} \mid \mathbf{I}\big)
$$

where $\hat{R}$ are detected text regions and $\hat{T}$ the recognized transcripts. Because this factorization is non-differentiable end-to-end, the error at each stage compounds multiplicatively rather than being jointly minimized — a region missed by $p_{\text{Det}}$ cannot be recovered downstream regardless of $p_{\text{NER}}$'s capacity.

Vision-Language Models (VLMs) collapse this factorization into a single conditional generation problem,


where $\mathbf{z}_{\text{vis}}(\mathbf{I})$ is a learned visual embedding and generation is autoregressive over a single unified parameter set $\theta$. This project evaluates **Qwen3-VL-2B-Instruct** for zero-shot ingredient extraction from food packaging images under this OCR-free paradigm.

---

## 2. Background: OCR as Constrained Sequence Transduction

Early OCR relied on handcrafted descriptors (HOG, projection profiles, connected components) brittle to font, illumination, and degradation variance. Deep OCR reframed the problem as **sequence transduction**: an image strip $\mathbf{x} \in \mathbb{R}^{H \times W}$ maps to a character sequence $\mathbf{y} = (y_1, \dots, y_L)$, $L \le W'$ (the downsampled feature width).

**CRNN + CTC** [1] removed the need for explicit per-character segmentation. A CNN backbone produces a feature sequence $\mathbf{h} \in \mathbb{R}^{T \times d}$, fed to a BiLSTM, and decoded via Connectionist Temporal Classification, which marginalizes over all alignments $\pi$ (paths over the extended alphabet $\Sigma \cup \{\text{blank}\}$) that collapse to the target via $\mathcal{B}(\pi) = \mathbf{y}$:

$$
p(\mathbf{y} \mid \mathbf{x}) = \sum_{\pi \,:\, \mathcal{B}(\pi) = \mathbf{y}} \prod_{t=1}^{T} p(\pi_t \mid \mathbf{x}), \qquad
\mathcal{L}_{\text{CTC}} = -\log p(\mathbf{y} \mid \mathbf{x})
$$

computed tractably via the forward-backward dynamic program in $O(T \lvert \mathbf{y} \rvert)$.

**TrOCR** [2] replaces the recurrent decoder with self-attention end-to-end: image patches are linearly projected ($\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot 3)} \to \mathbf{E}\mathbf{x}_p$) and processed by a ViT (DeiT-initialized) encoder; a RoBERTa/UniLM-initialized Transformer decoder generates wordpieces autoregressively using standard scaled dot-product attention,

$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

with cross-attention conditioning each decoder layer on the full encoder output sequence rather than a single fixed-length feature vector — directly removing the BiLSTM's fixed-context bottleneck.

OCR remains, fundamentally, a **maximum-likelihood transcription** objective: it recovers all visible text irrespective of semantic relevance, leaving entity discrimination to downstream heuristics.

<p align="center">

<br></b> <img width="702" height="391" alt="Screenshot 2026-06-30 at 4 40 11 PM" src="https://github.com/user-attachments/assets/0d2b8a49-9f09-4659-a0d8-3b16b1cdd181" />
</p>

---

## 3. OCR-based KIE Pipeline

$$
\textbf{Image} \;\to\; \text{Preprocess} \;\to\; \underbrace{\text{Detect}}_{\hat R} \;\to\; \underbrace{\text{Recognize}}_{\hat T} \;\to\; \underbrace{\text{NER / Rules}}_{\mathcal{E}} \;\to\; \textbf{Structured Output}
$$

**CRAFT** [3] localizes text at the *character* level rather than the word/line level, predicting two pixel-wise heatmaps: a region score $S_r(p) = \exp\!\left(-\frac{\lVert p - p_c \rVert^2}{2\sigma^2}\right)$ (a 2D isotropic Gaussian centered at each character box, perspective-warped onto the box) and an affinity score $S_a(p)$ over the diagonal regions linking adjacent characters — letting arbitrary-length words be reconstructed via simple connected-component analysis on the thresholded maps. Training minimizes pixel-wise MSE against these synthetically-generated Gaussian targets.

**DBNet** [4] instead predicts a probability map $P$ and threshold map $T$ jointly, replacing the non-differentiable binarization step $B_{i,j} = \mathbb{1}[P_{i,j} \ge t]$ with a **Differentiable Binarization (DB)** surrogate:

$$
\hat{B}_{i,j} = \frac{1}{1 + e^{-k(P_{i,j} - T_{i,j})}}
$$

with steepness $k$ (typically 50), allowing the binarization threshold itself to be learned via backpropagation rather than fixed post-hoc — this is what gives DBNet its real-time speed/accuracy trade-off.

Limitations of this pipeline: (i) compounding error — $\text{CER}_{\text{final}} \gtrsim \text{CER}_{\text{detect}} + \text{CER}_{\text{recognize}}$ under independence assumptions; (ii) spatial/visual semantics are discarded once $\hat T$ is produced, so the NER stage operates on a 1-D token stream stripped of layout context. EasyOCR, PaddleOCR, MMOCR, and TrOCR were benchmarked during this internship to characterize these failure modes empirically.

<p align="center">
<img src="https://github.com/clovaai/CRAFT-pytorch/raw/master/figures/craft_example.gif" width="500" alt="CRAFT character-region detection">
<br><b>Figure 2.</b> CRAFT region/affinity heatmaps reconstructing word-level boxes from character-level Gaussians. Source: <a href="https://github.com/clovaai/CRAFT-pytorch">clovaai/CRAFT-pytorch</a>.
</p>

---

## 4. Motivation for VLMs

Food packaging images contain dense, heterogeneous text — multilingual labels, nutrition tables, marketing claims, decorative typography — where the *ingredient list* is one semantic subset among many. OCR recovers $\hat T$ in full; KIE requires a selection function $\phi: \hat T \to \mathcal{E}$ that conventional pipelines approximate with brittle regex/NER heuristics.

Contrastive vision-language pretraining provides the representational substrate for this selection to be learned implicitly. **CLIP** [5] maximizes agreement between matched image-text pairs $(\mathbf{I}_i, \mathbf{t}_i)$ in a batch of size $N$ via the symmetric InfoNCE objective:

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^N \left[
\log \frac{e^{\, \text{sim}(\mathbf{v}_i, \mathbf{l}_i)/\tau}}{\sum_{j} e^{\, \text{sim}(\mathbf{v}_i, \mathbf{l}_j)/\tau}}
+ \log \frac{e^{\, \text{sim}(\mathbf{v}_i, \mathbf{l}_i)/\tau}}{\sum_{j} e^{\, \text{sim}(\mathbf{v}_j, \mathbf{l}_i)/\tau}}
\right]
$$

where $\text{sim}(\mathbf{v},\mathbf{l}) = \mathbf{v}^{\top}\mathbf{l} / \lVert \mathbf{v}\rVert \lVert \mathbf{l} \rVert$ and $\tau$ is a learned temperature. This requires the full $N \times N$ softmax normalization across the batch, which is memory- and batch-size-sensitive.

**SigLIP** [6] replaces this with an independent per-pair sigmoid loss, removing the need for global normalization:

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^N \log \sigma\!\left( z_{ij}\big(t \cdot \text{sim}(\mathbf{v}_i,\mathbf{l}_j) + b\big) \right), \quad z_{ij} = \begin{cases}+1 & i=j \\ -1 & i \ne j\end{cases}
$$

with learnable scale $t$ and bias $b$. Because each pair is treated as an independent binary classification problem, SigLIP decouples loss computation from batch size, enabling stable training at far smaller per-device batches — a key reason it was adopted as Qwen3-VL's vision tower (SigLIP2).

LLaVA, Qwen2-VL, and Qwen3-VL [7–9] extend this contrastively-pretrained visual representation into autoregressive instruction-following, replacing *"what text exists?"* with *"what are the ingredients?"* — letting irrelevant regions be implicitly suppressed during decoding rather than explicitly filtered post-hoc.

---

## 5. Qwen3-VL Architecture

Qwen3-VL is a three-module system: a **SigLIP2 ViT** vision encoder, an **MLP-based vision-language merger**, and a **Qwen3 decoder-only LLM** [9].

**Patch embedding.** The image is partitioned into non-overlapping $P \times P$ patches, flattened, and linearly projected:

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \, \mathbf{E}\mathbf{x}_p^1; \, \mathbf{E}\mathbf{x}_p^2; \, \dots; \, \mathbf{E}\mathbf{x}_p^N] + \mathbf{E}_{\text{pos}}, \qquad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot 3) \times d}
$$

**Naive Dynamic Resolution (NDR)** removes the fixed-resolution constraint of standard ViTs: $N = \frac{HW}{P^2}$ varies per image (with a configurable token budget, e.g. 256–1280 visual tokens for a single image at 32× spatial compression), preserving native aspect ratio and avoiding the information loss of forced square resizing — critical for elongated packaging labels with dense small-font ingredient text.

**DeepStack** fuses intermediate ViT layer features (not just the final layer) into the LLM input stream, injecting multi-scale visual detail directly rather than relying solely on the last encoder layer's receptive field — improving fine-grained character-level legibility for OCR-adjacent tasks.

**Interleaved M-RoPE** extends rotary position embeddings to three axes (temporal $t$, height $h$, width $w$) by partitioning the rotation frequency bands across axes instead of concatenating raw position indices:

$$
\text{M-RoPE}(\mathbf{x}, (t,h,w)) = R_\Theta(t) \oplus R_\Theta(h) \oplus R_\Theta(w)
$$

with frequency bands interleaved (rather than block-partitioned as in Qwen2-VL) so every rotation sub-block sees a roughly uniform mix of temporal/spatial frequencies — improving long-horizon spatial generalization, relevant when reasoning jointly over an ingredient table's row/column layout.

**Generation.** The decoder is trained with standard next-token cross-entropy over interleaved visual and textual tokens:

$$
\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log p_\theta\big(y_t \mid y_{<t}, \mathbf{z}_{\text{vis}}\big)
$$

with no auxiliary OCR-specific loss term — the model is never explicitly supervised to transcribe; ingredient extraction emerges purely from instruction-tuned generation.

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_arc.jpg" width="850" alt="Qwen3-VL architecture: Interleaved-MRoPE and DeepStack">
<br><b>Figure 3.</b> Qwen3-VL architecture updates — Interleaved-MRoPE and DeepStack feature fusion. Source: <a href="https://github.com/QwenLM/Qwen3-VL">QwenLM/Qwen3-VL</a>, <a href="https://arxiv.org/pdf/2511.21631">arXiv:2511.21631</a>.
</p>

---

## 6. Inference Pipeline

$$
\textbf{Image} \to \text{AutoProcessor} \to \text{SigLIP2 Encoder} \to \mathbf{z}_{\text{vis}} \to \text{Projector} \to \text{Qwen3 Decoder} \to \text{Autoregressive Generation} \to \textbf{Ingredient List}
$$

`AutoProcessor` performs dynamic resizing under the NDR pixel budget, normalization $(\mathbf{x} - \mu)/\sigma$, prompt tokenization, and chat-template assembly. Inference ran with `Qwen3VLForConditionalGeneration` in `bfloat16` with automatic device placement; no detection, cropping, or recognition stage precedes generation — the full image is consumed in one forward pass, and $\phi$ (relevance selection) is computed implicitly inside the attention layers conditioned on the natural-language instruction.

---

## 7. Dataset — Ingredients_200

187 product packaging images with manually verified ground truth, each comprising: package image, ground-truth ingredient sequence, OCR JSON annotations (where available), and human-verified annotations. Labels were normalized for punctuation, casing, and tokenization consistency to minimize annotation noise during lexical scoring.

<p align="center">
<img src="https://raw.githubusercontent.com/clovaai/deep-text-recognition-benchmark/master/figures/STR_examples.png" width="700" alt="Representative scene-text / packaging dataset samples">
<br><b>Figure 4.</b> Representative variation in typography, lighting, and layout across packaging-style text recognition benchmarks. Source: <a href="https://github.com/clovaai/deep-text-recognition-benchmark">clovaai/deep-text-recognition-benchmark</a>.
</p>

---

## 8. Evaluation Protocol

Predictions and references were normalized (case-folding, whitespace, punctuation) prior to scoring.

**Character/Word Error Rate** — normalized Levenshtein edit distance over the operation set $\{$substitution $S$, deletion $D$, insertion 

$$
\text{CER} = \frac{S_c + D_c + I_c}{N_c}, \qquad \text{WER} = \frac{S_w + D_w + I_w}{N_w}
$$

where $N_c, N_w$ are reference character/word counts. **These are misleading for KIE**: a prediction with every ingredient correct but in different list order, or with a synonym substitution ("turmeric" vs "haldi"), incurs high WER despite being semantically correct — WER is a sequence-alignment metric, not a set-membership metric.

**Set-based metrics**, given predicted set $\hat{\mathcal{E}}$ and reference set $\mathcal{E}$:

$$
P = \frac{|\hat{\mathcal{E}} \cap \mathcal{E}|}{|\hat{\mathcal{E}}|}, \qquad
R = \frac{|\hat{\mathcal{E}} \cap \mathcal{E}|}{|\mathcal{E}|}, \qquad
F_1 = \frac{2PR}{P+R}, \qquad
J = \frac{|\hat{\mathcal{E}} \cap \mathcal{E}|}{|\hat{\mathcal{E}} \cup \mathcal{E}|}
$$

**RapidFuzz similarity** uses normalized Indel distance for approximate token matching, tolerant to minor OCR-style spelling noise:

$$
\text{sim}_{\text{fuzz}}(a,b) = 1 - \frac{\text{lev}(a,b)}{|a|+|b|}
$$

The central empirical finding motivating this paper's framing: WER/CER systematically *under-rate* semantically correct VLM extractions (penalizing reordering, paraphrase, and granularity mismatch) relative to $F_1$/Jaccard, which is why standard KIE benchmarking practice privileges set-based scoring over raw transcription accuracy — confirmed across the 187-image Qwen3-VL-4B-Instruct inference run.

---

## 9. Implementation

Built on **Hugging Face Transformers** + **PyTorch**, using `Qwen3VLForConditionalGeneration` with the corresponding `AutoProcessor`; inference in `bfloat16` with automatic device placement. The pipeline supports batch processing, checkpoint-based resumption, structured JSON prediction storage, and crash recovery. Evaluation is decoupled from inference for independent benchmarking across OCR and VLM systems under one protocol.

```text
.
├── dataset/                  # Ingredients_200 dataset
├── outputs/                  # Generated predictions
├── evaluation/               # Evaluation scripts
├── config.py                 # Model and dataset configuration
├── main.py                   # End-to-end inference pipeline
├── evaluate.py                # Quantitative evaluation
└── README.md
```

---

## 10. Observations

The dominant effect is the shift from **transcription-accuracy maximization** to **task-conditioned semantic generation**: maximizing $p(\mathbf{y} \mid \mathbf{x})$ over the full text surface (OCR) versus maximizing $p(\mathcal{E} \mid \mathbf{I}, \text{instruction})$ over a task-relevant subset (VLM-KIE). The latter implicitly performs entity selection $\phi$ inside the attention mechanism, conditioned jointly on visual layout and the natural-language instruction, rather than as a separate post-hoc NER stage operating on a flattened token stream stripped of spatial context.

The cost is inference-time compute: a 2–4B parameter autoregressive VLM forward pass is substantially more expensive than a CRNN/CTC or DBNet forward pass. The benefit is robustness to the layout heterogeneity (multilingual labels, nutrition tables, decorative fonts) that breaks rule-based downstream filtering in conventional pipelines.

---

## 11. Future Work

- LoRA/QLoRA parameter-efficient fine-tuning of Qwen3-VL for domain-specific ingredient extraction.
- Scaling-law analysis across Qwen3-VL-2B/4B/8B on KIE set-based metrics.
- Multilingual packaging and cross-domain document understanding.
- Constrained decoding for structured JSON generation (grammar-constrained sampling over a fixed schema).
- Benchmarking OCR-free VLM pipelines against document foundation models on ICDAR-standard KIE tasks.

---

## References

[1] Shi, B., Bai, X., & Yao, C. *An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN).* IEEE TPAMI, 2017.
[2] Li, M. et al. *TrOCR: Transformer-based OCR with Pre-trained Models.* AAAI, 2023.
[3] Baek, Y. et al. *CRAFT: Character Region Awareness for Text Detection.* CVPR, 2019.
[4] Liao, M. et al. *Real-Time Scene Text Detection with Differentiable Binarization.* AAAI, 2020.
[5] Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML, 2021.
[6] Zhai, X. et al. *Sigmoid Loss for Language Image Pre-Training (SigLIP).* ICCV, 2023.
[7] Liu, H. et al. *Visual Instruction Tuning (LLaVA).* NeurIPS, 2023.
[8] Wang, P. et al. *Qwen2-VL Technical Report.* arXiv:2409.12191, 2024.
[9] Qwen Team. *Qwen3-VL Technical Report.* arXiv:2511.21631, 2025.

---

## Acknowledgements

Carried out during my Summer Research Internship at the **Indian Institute of Technology Ropar** under **Dr. Gyanendra Singh**, investigating Vision-Language Models for OCR-free Key Information Extraction and the transition from modular OCR pipelines to unified multimodal document intelligence.
