# Raíces

**Evaluating zero-shot multimodal models for culturally-specific visual understanding.**

*An applied data science project exploring how well foundation models (CLIP, DINOv2, Llama-3) capture culturally-situated visual concepts, and where they systematically fail.*

---

## Motivation

Foundation vision-language models are trained on internet-scale corpora that are demonstrably skewed toward Western, English-speaking contexts. This project asks a concrete question:

> **How reliably can off-the-shelf multimodal embeddings identify culturally-specific visual motifs, and what does the error distribution look like across cultures, categories, and image conditions?**

Most evaluation benchmarks for CLIP and similar models measure performance on ImageNet-style object recognition. Very few measure performance on culturally-situated concepts (e.g., *papel picado*, *ofrenda*, *huipil*, *mate gourd*) where a model's training-data coverage directly shapes whose heritage it can "see." Raíces treats this as a measurement problem first and a product second.

The applied layer (generating multilingual heritage narratives from detected motifs) is the vehicle for surfacing model behavior to a non-technical audience. The underlying contribution is the dataset, taxonomy, and evaluation framework.

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
A hand-curated taxonomy of cultural motifs organized by tradition, category (object / setting / food / clothing / activity), and sub-culture. Each motif has 2 to 3 prompt variants for robustness testing. Scope for v1: Latin American heritage (63 motifs across 11 traditions).

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

## Taxonomy v1 composition

63 motifs, 189 prompt variants, 11 traditions.

**By tradition:**

| Tradition             | Motifs |
|-----------------------|-------:|
| Mexican               |     17 |
| Peruvian              |      9 |
| Argentine             |      6 |
| Brazilian             |      6 |
| Colombian             |      5 |
| Guatemalan            |      4 |
| Cuban                 |      4 |
| Andean (pan-regional) |      4 |
| Bolivian              |      3 |
| Puerto Rican          |      3 |
| Chilean               |      2 |

**By category:**

| Category | Motifs |
|----------|-------:|
| Object   |     23 |
| Activity |     13 |
| Clothing |     11 |
| Setting  |      8 |
| Food     |      8 |

The distribution is intentionally uneven and reflects the long tail of personal familiarity and source-material availability. Mexican coverage is deepest; several traditions are sparse. **Per-culture macro-F1 (rather than micro-F1) is the headline metric** so this imbalance does not get hidden in the average.

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

---

## Scope, honestly

v1 covers **Latin American heritage only**, with deepest coverage of Mexican motifs and thin coverage of several other traditions (see composition table). No claim is made about "global cultural understanding." That claim would require an order-of-magnitude larger taxonomy and meaningful contribution from each represented community.

The taxonomy is treated as a political artifact, not a neutral list. What it includes and excludes shapes whose heritage the system can recognize. Expansion happens through community review, not through me adding motifs from Wikipedia.

---

## Ethical considerations

- **Representation, not identification.** The system reports motifs as *observations about an image*, never as identity claims about people in the image. A detected huipil indicates "a huipil is visible," not "this person is Indigenous Mexican."
- **Anti-stereotyping in narratives.** The narrative prompt forbids invention of cultural backstory beyond detected motifs, and factuality evaluation explicitly measures adherence to this constraint.
- **Consent for personal photos.** All photos used in development and any public demo require explicit consent from the people pictured. Public examples use Creative Commons or self-provided imagery.
- **Privacy via on-device inference.** The Core ML variant exists so family photos never leave the device, a product claim backed by architecture, not marketing.
- **Sensitive practices (e.g., coca leaf rituals).** Some motifs involve practices that have been mischaracterized in non-source-community contexts. Annotation guidance and prompts emphasize ceremonial framing; reviewers from source communities are the final arbiters.