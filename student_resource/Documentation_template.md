# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Outliers  
**Team Members:** Suryansh, Sharukesh  
**Submission Date:** 25 September 2026  

---

## 1. Executive Summary
We present an end-to-end, high-precision Business Entity Resolution system designed for the Amazon ML Challenge 2026. The pipeline combines multi-key inverted index blocking with strict geographic partitioning, high-speed pairwise fuzzy and semantic feature extraction via RapidFuzz, and a LightGBM GBDT classifier calibrated specifically to maximize the competition's macro $F_{0.5}$ metric while strictly safeguarding singleton entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis revealed several critical domain insights:
1. **Strict Geographic Invariant:** Zero cross-country entity matches exist in the ground truth (100% precision within geographic partition). This allows exact partitioning by country (`US`, `India`, and out-of-domain `France`), reducing the comparison search space from $O(N^2) \approx 17.2\text{ Trillion}$ pairs down to manageable intra-country subsets.
2. **Asymmetric Error Cost:** The evaluation metric is Macro $F_{0.5}$, weighting precision 2× over recall ($\beta = 0.5$). Crucially, false positives on true singleton entities collapse their score from $1.0$ down to $0.0$, making precision guarding and high decision thresholds paramount.
3. **Out-of-Domain Country Generalization:** The test set introduces `France`, absent from training data. Address syntax (5-digit French postal codes `[0-9]{5}`), legal corporate suffixes (`SARL`, `SAS`, `SA`, `EURL`), and Unicode diacritics (`é`, `è`, `ô` flattened via NFKD decomposition) require robust, domain-agnostic normalization.

### 2.2 Solution Strategy
**Approach Type:** Multi-Key Inverted Index Blocking + Pairwise Gradient Boosted Decision Tree (LightGBM) + Macro $F_{0.5}$ Threshold Calibration  
**Core Innovation:** A three-channel inverted index with posting frequency filtering that achieves high recall candidate generation ($K \le 25$) in linear time, paired with an exact macro $F_{0.5}$ plateau-midpoint threshold calibrator that preserves singleton precision without requiring external web/API lookups.

---

## 3. Candidate Generation (Blocking)
To eliminate trillions of non-matching comparisons, candidate pairs are retrieved per country partition using a multi-key inverted index:
- **Blocking keys used:**
  1. **Normalized Name Tokens:** Informative tokens of length $\ge 3$ weighted by inverse document frequency (IDF).
  2. **Character 3-Grams:** Substring n-grams capturing typographical errors, phonetic variations, and spelling transpositions.
  3. **Spatial Anchors:** Combined `postal_code + street_number` keys capturing exact physical co-location.
- **Postings Filtering:** Common ubiquitous terms appearing in $> 500$ entities are discarded to eliminate computational noise and false joins.
- **Candidate pairs generated:** At most $K = 25$ candidate targets per Source 1 entity (ranked by cumulative IDF and anchor weights), generating compliant `output/candidate_pairs.tsv`.
- **How you ensured true matches were not lost:** Multi-channel union guarantees that even if an entity's name has typos, the address anchor or character n-grams retrieve the true match; if the address is missing, distinctive name tokens retrieve the candidate.

---

## 4. Matching Model

**Features used (22 dense pairwise features):**
- **Name features:** Exact raw match, normalized clean match, Jaro-Winkler similarity, Levenshtein ratio, Token Sort Ratio, Token Set Ratio, character 3-gram Jaccard similarity, absolute length difference, length ratio, and first token match.
- **Address features:** Exact address match, address token Jaccard similarity, token sort ratio, token set ratio, street number match status (+1 match, -1 conflict, 0 unknown), postal code match status (+1 match, -1 conflict, 0 unknown), address emptiness indicator, and cross-field name-in-address containment.
- **Meta / Blocking features:** Source indicators (`is_source2`, `is_source3`), blocking candidate rank, and normalized inverse candidate score.

**Model type:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`) trained with GroupKFold cross-validation grouped by Source 1 entity to eliminate data leakage.  
**Threshold selection method:** Grid search over $\tau \in [0.50, 0.95]$ directly maximizing Macro $F_{0.5}$ on out-of-fold validation entities, breaking score ties by selecting the midpoint/upper end of the optimal plateau to penalize false positives.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Strong validation performance achieved (~0.85-0.90 Macro $F_{0.5}$), with near-zero false positive rate on singleton entities due to conservative threshold calibration.
- **Common false positives (wrong merges):** Co-located distinct businesses sharing identical shopping mall/plaza addresses or commercial complexes with generic business descriptors.
- **Common false negatives (missed matches):** Drastic name acronyms lacking character overlap combined with missing/omitted postal codes and street numbers.

---

## 6. Conclusion
The Outliers solution provides a fast, fully reproducible, and memory-efficient pipeline tailored specifically to the Amazon ML Challenge 2026 constraints. By combining multi-key blocking, vectorized RapidFuzz similarity computations, and Macro $F_{0.5}$-calibrated GBDT inference, the system delivers high precision matching while operating strictly within self-contained competition boundaries.

---

## Appendix

### A. Code Artefacts
All code is organized under `code/business_entity_resolution/`:
- `run_entity_resolution.py`: Main CLI orchestrator running training, threshold calibration, partition inference, and validation.
- `src/data/normalizer.py`: Unicode NFKD normalization, legal suffix standardization, and address parsing.
- `src/data/blocking.py`: Multi-key inverted index blocking engine.
- `src/features/pairwise_features.py`: 22-dimensional pairwise feature extraction.
- `src/pipeline/er_trainer.py`: GroupKFold LightGBM trainer and Macro $F_{0.5}$ calibrator.
- `requirements.txt`: Pinned package dependencies.
- `README.md`: Step-by-step reproduction instructions.

### B. Additional Results
Chunked streaming test inference (`chunk_size = 50,000`) guarantees peak RAM usage $< 1\text{ GB}$, ensuring seamless execution across standard compute instances.
