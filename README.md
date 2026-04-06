# PoliMemeDecode Challenge - Team Backprop Sust

**Final Standings:** **Private Leaderboard:** 20th out of 151 teams
**Public Leaderboard:** 37th out of 151 teams

This repository contains our solution for the **PoliMemeDecode** National Datathon, organized by the Department of CSE, CUET, as part of CUET CSE FEST 2025. 

---

## Detailed Methodology
**To review our complete approach, please refer to this document: [DOCUMENTATION.pdf](./DOCUMENTATION.pdf)**

---

## Competition Overview
PoliMemeDecode challenges participants to build machine learning models capable of classifying meme content into **Political** and **Non-Political** categories. The challenge emphasizes understanding the subtle intent, satire, and commentary within memes, moving beyond mere humor detection. 

The evaluation metric for this competition is the **Macro F1 Score**, ensuring balanced performance across both classes. A key difficulty of this challenge is a distribution-shifted test dataset, necessitating creative preprocessing and robust modeling strategies.

## Our Approach
To capture the semantics of political memes, we treated each sample as a multi-view text object and implemented a dual-stage classification pipeline:

### 1. Data Acquisition & Semantic Enrichment
* **OCR:** Extracted raw text from memes using HunyuanOCR.
* **Semantic Enrichment:** Utilized the Gemma Multimodal Inference model to generate semantic descriptions by feeding it both the original image and OCR text. We extracted three features: `text_meaning`, `combined_meaning`, and `context_analysis`.
* **Synthetic Logo Dataset:** Created a background dataset of 20,000 images with randomly augmented political logos to simulate internet meme noise.

### 2. Multi-Stage Pipeline
* **Dual-Encoder Text Classifier:** A custom architecture leveraging `microsoft/deberta-v3-large` and `RoBERTa-Large`. Features interact via a Multi-Head Cross-Attention mechanism to resolve ambiguities like sarcasm.
* **Visual Heuristics (Post-Processing):**
  * **Logo Recognition:** A YOLOv8-Nano model fine-tuned to detect political logos.
  * **Face Recognition:** An InsightFace pipeline using the `buffalo_l` model to detect prominent politicians via cosine similarity.
  * **Keyword Filtering:** Regex scanning for specific political entities in Bengali and English.

## Repository Structure
* `DOCUMENTATION.pdf` - Detailed breakdown of our methodology, architecture, and results.
* `FINAL_INFERENCE_NOTEBOOK.ipynb` - The final notebook used to generate our test set submissions.
* `cuet-cse-datathon main training pipeline.ipynb` - The core training pipeline for our dual-encoder model.
* `faces recognition.zip` - Assets and scripts related to our InsightFace facial recognition pipeline.
* `logo recognition.zip` - Assets and scripts related to our YOLOv8 logo detection pipeline.
