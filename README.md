# Raíces

**Evaluating zero-shot multimodal models for culturally-specific visual understanding.**

*An applied data science project exploring how well foundation models (CLIP, DINOv2, Llama-3) capture culturally-situated visual concepts, and where they systematically fail.*

---

## Motivation

Foundation vision-language models are trained on internet-scale corpora that are demonstrably skewed toward Western, English-speaking contexts. This project asks a concrete question:

> **How reliably can off-the-shelf multimodal embeddings identify culturally-specific visual motifs — and what does the error distribution look like across cultures, categories, and image conditions?**

Most evaluation benchmarks for CLIP and similar models measure performance on ImageNet-style object recognition. Very few measure performance on culturally-situated concepts (e.g., *papel picado*, *ofrenda*, *huipil*, *mate gourd*) where a model's training-data coverage directly shapes whose heritage it can "see." Raíces treats this as a measurement problem first and a product second.

The applied layer — generating multilingual heritage narratives from detected motifs — is the vehicle for surfacing model behavior to a non-technical audience. The underlying contribution is the dataset, taxonomy, and evaluation framework.

---

## Research questions

1. **Coverage.** Given a curated taxonomy of N cultural motifs across K traditions, what fraction are detectable by zero-shot CLIP above a usable confidence threshold?
2. **Calibration.** Are CLIP confidence scores well-calibrated across cultural categories, or does the model express higher confidence on over-represented cultures?
3. **Prompt sensitivity.** How much does detection accuracy vary with prompt phrasing (e.g., native-language terms vs. English descriptions vs. paraphrases)?
4. **Cross-lingual narrative fidelity.** When Llama-3 generates narratives in non-English languages, does factual fidelity to detected motifs degrade? By how much, and for which languages?
5. **Hallucination rate.** How often does the narrative generator invent cultural details not present in the detected-motif input, and what guardrails reduce this?

---

## Approach

### 1. Taxonomy construction
A hand-curated taxonomy of cultural motifs organized by tradition, category (object / setting / food / clothing / activity), and sub-culture. Each motif has 2–3 prompt variants for robustness testing. Scope for v1: Latin American heritage (~70 motifs); scope grows with community contribution.

### 2. Evaluation dataset
A hand-labeled image set of ~500 photos sourced from Creative Commons and consented personal submissions, with multi-label motif annotations, scene labels, and emotional-tone labels. Each photo has provenance metadata (source, cultural context note from contributor).

### 3. Detection pipeline
- **CLIP ViT-L/14** (or MobileCLIP for the on-device variant) for zero-shot motif detection via cosine similarity against taxonomy prompts.
- **DINOv2** for scene-level features and nearest-neighbor scene tagging.
- **Llama-3.2 3B-Instruct** for narrative synthesis, conditioned on detected motifs + tone + scene context with explicit anti-hallucination system prompting.

### 4. Evaluation
- **Detection:** precision@k, recall@k, per-culture macro-F1, calibration curves.
- **Narrative:** automated factuality scoring (does the narrative only mention motifs present in input?), BLEU/chrF for multilingual consistency against a reference set, human ratings on a 5-point cultural-fidelity scale from native-speaker reviewers.
- **Ablations:** CLIP-only vs. CLIP+DINOv2, prompt-variant sensitivity, impact of quantization on on-device variant.

---

## Tech stack

| Layer | Tool |
|---|---|
| Vision embeddings | `open_clip_torch`, `transformers` (DINOv2) |
| Language model | `transformers` + `meta-llama/Llama-3.2-3B-Instruct` |
| Data handling | `pandas`, `datasets`, `pillow` |
| Evaluation | `scikit-learn`, `sacrebleu`, custom factuality checks |
| Demo | `gradio` on Hugging Face Spaces |
| On-device (optional) | `coremltools`, MLX, SwiftUI |