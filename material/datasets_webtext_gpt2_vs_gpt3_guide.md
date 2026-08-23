# Pretraining Datasets Guide: WebText (GPT-2), GPT-3, RedPajama, SlimPajama & FineWeb

A comprehensive architectural and engineering guide on the pretraining datasets used across the major generations of Large Language Models:
* **GPT-2 (WebText)**: Social-curation heuristics (Reddit karma).
* **GPT-3 (300B Mixture)**: Classifier-filtered Common Crawl + high-quality source oversampling.
* **RedPajama-1T (Together AI)**: Open-source reproduction of Meta's LLaMA dataset recipe.
* **SlimPajama-627B (Cerebras)**: Large-scale MinHashLSH deduplicated and cleaned distillation of RedPajama.
* **RedPajama-V2 (30T Tokens)**: Trillion-scale open corpus with 40+ computable quality annotations.
* **FineWeb & FineWeb-Edu (Hugging Face)**: Modern synthetic LLM-as-a-judge quality scoring.

---

## 1. Overview & Evolutionary Timeline of Pretraining Corpora

Language modeling transitioned from small, human-annotated academic benchmarks to web-scale uncurated corpora, and finally to heavily deduplicated, classifier-filtered datasets.

| Model / Dataset | Year | Raw Volume | Training Tokens | Key Philosophy & Filtering Mechanism |
|:---|:---:|:---|:---:|:---|
| **GPT-1 (BookCorpus)** | 2018 | ~4.5 GB | ~1 Billion | Long contiguous spans of structured narrative text. |
| **GPT-2 (WebText)** | 2019 | ~40 GB (8M docs) | ~10 Billion | Human-curated web links via Reddit karma threshold ($\ge 3$). |
| **GPT-3 (Mixture)** | 2020 | ~570 GB (post-filter) | ~300 Billion | 5-source blend, logistic regression quality classifier, MinHash deduplication. |
| **RedPajama-1T** | 2023 | ~5 TB | ~1.21 Trillion | Open reproduction of Meta LLaMA recipe across 7 data slices. |
| **SlimPajama (Cerebras)** | 2023 | ~2.5 TB | ~627 Billion | Extensively deduplicated RedPajama (removed 49.6% duplicated bytes via distributed MinHashLSH). |
| **RedPajama-V2** | 2023 | ~100 TB | ~30 Trillion | 84 Common Crawl dumps pre-computed with 40+ quality signals. |
| **FineWeb-Edu** | 2024 | ~44 TB | 1.3+ Trillion | Scored & filtered with Llama-3-70B-Instruct educational quality classifier. |

---

## 2. GPT-2 Training Dataset: WebText

### Motivation & Design
OpenAI avoided raw web scrapes (like unfiltered Common Crawl) for GPT-2 because raw web crawls are plagued with spam, machine-translated text, SEO boilerplates, pornography, and low-quality content.

To obtain high-quality human-curated content without manual annotation, OpenAI used **social curation as an implicit quality signal**:
* Scraped all outbound links from **Reddit** posted up to **December 2017**.
* Only included links that received at least **3 karma** (upvotes minus downvotes), heuristic proxy for human approval and heuristic quality.

```
                  ┌────────────────────────────────────────┐
                  │    Reddit Submissions (≤ Dec 2017)     │
                  └───────────────────┬────────────────────┘
                                      │ Filter: Karma ≥ 3
                                      ▼
                  ┌────────────────────────────────────────┐
                  │          Outbound Web URLs             │
                  └───────────────────┬────────────────────┘
                                      │ HTML Extraction (Dragnet / Newspaper)
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Deduplication & Wikipedia Removal      │
                  └───────────────────┬────────────────────┘
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ WebText: 40 GB, 8M docs, ~10B Tokens   │
                  └────────────────────────────────────────┘
```

### Key Characteristics of WebText
1. **Size & Scope:**
   - **40 GB** of clean text across **~8 million documents**.
   - Encoded to approximately **8–10 billion tokens** with the GPT-2 BPE tokenizer ($V = 50,257$).
2. **Exclusions & Contamination Mitigation:**
   - All **Wikipedia pages were explicitly removed** from WebText. OpenAI did this because Wikipedia articles are heavily used in downstream evaluation benchmarks; removing Wikipedia prevented trivial evaluation memorization.
3. **Open-Source Reproductions:**
   - OpenAI never publicly released the raw WebText corpus due to copyright and licensing constraints.
   - Open reproductions:
     - **OpenWebText (2019):** Scrapes Reddit submissions through 2018 with $\ge 3$ karma (~38 GB text, ~9B tokens).
     - **OpenWebText2 (EleutherAI, 2020):** Extended Reddit scraping to 2020, added cleaner deduplication and content filters for The Pile.

---

## 3. GPT-3 Training Dataset: The 300B-Token Mixture

### Why WebText Was Insufficient for GPT-3
For the 175B-parameter GPT-3 model, training for 300 billion tokens on the 10B-token WebText would have meant repeating the same data 30 times, causing catastrophic overfitting and performance plateaus. OpenAI therefore expanded to web-scale **Common Crawl**, introducing an extensive 3-stage cleaning pipeline.

### The 5 Datasets in the GPT-3 Mixture

| Dataset Source | Raw Pre-filter Tokens | Cleaned Tokens in Pool | Weight in Training Mixture | Epochs Elapsed in 300B Tokens |
|:---|:---:|:---:|:---:|:---:|
| **Filtered Common Crawl** | ~1.0 Trillion | 410 Billion | **60%** | 0.44 (44% seen once) |
| **WebText2** | 19 Billion | 19 Billion | **22%** | 3.44 |
| **Books1** | 12 Billion | 12 Billion | **8%** | 2.00 |
| **Books2** | 55 Billion | 55 Billion | **8%** | 0.44 |
| **Wikipedia (English)** | 3 Billion | 3 Billion | **3%** | 3.40 |
| **Total Mixture** | — | **~500 Billion** | **100%** | **300 Billion sampled** |

```
                       GPT-3 TRAINING DATASET MIXTURE
                     ┌────────────────────────────────┐
                     │   Common Crawl (Filtered) 60%  │
                     ├────────────────────────────────┤
                     │   WebText2                 22% │
                     ├────────────────────────────────┤
                     │   Books1                    8% │
                     ├────────────────────────────────┤
                     │   Books2                    8% │
                     ├────────────────────────────────┤
                     │   Wikipedia (English)       3% │
                     └────────────────────────────────┘
```

> [!IMPORTANT]
> **Oversampling High-Quality Data Sources:**
> High-quality datasets (WebText2, Wikipedia, Books1) were intentionally **oversampled** during training. Even though Wikipedia represents less than 1% of the raw available tokens, it was assigned a **3% weight** in the sampling distribution, meaning it was cycled through ~3.4 times.

---

## 4. GPT-3 Data Processing & Filtering Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GPT-3 COMMON CRAWL FILTERING PIPELINE                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  [Stage 1: Logistic Regression Quality Classifier]
  • Positive Examples: WebText, Wikipedia, Books (Known High Quality)
  • Negative Examples: Unfiltered Common Crawl
  • Features: Document word counts, punctuation stats, n-gram perplexity
  • Keep probability: p = sigmoid(score) -> downsamples low-quality pages
                                      │
                                      ▼
  [Stage 2: Fuzzy Deduplication via MinHashLSH]
  • 13-gram Jaccard similarity across documents
  • Removed duplicates, syndicated articles, mirrors, and templated boilerplates
  • Removed documents with high overlap with high-quality reference sets
                                      │
                                      ▼
  [Stage 3: Benchmark Contamination Filtering]
  • 13-gram exact match search against all test/dev sets (LAMBADA, SuperGLUE, ARC)
  • Removed overlaps or documented contaminated splits in the appendix
                                      │
                                      ▼
  Result: 410 Billion Clean Tokens from >45 Terabytes of Raw Crawl
```

---

## 5. RedPajama & SlimPajama: The Open Foundation Era

### A. RedPajama-1T (Together AI, April 2023)
When Meta released LLaMA 1 in early 2023, its high quality derived from a curated 1.4T token recipe across 7 distinct domains. However, Meta did not release the underlying dataset. 

**Together AI** (along with Ontocord.ai, ETH DS3Lab, Stanford CRFM, and Hazy Research) launched the **RedPajama project** to create a 100% open-source reproduction of the LLaMA pretraining dataset (~1.21 Trillion tokens).

```
                      REDPAJAMA-1T COMPOSITION (1.21T Tokens)
┌───────────────────────────┬──────────────┬──────────────────────────────────┐
│ Slice                     │ Token Volume │ Source / Description             │
├───────────────────────────┼──────────────┼──────────────────────────────────┤
│ Common Crawl              │ 878 Billion  │ 5 CC dumps filtered via CCNet    │
│ C4                        │ 175 Billion  │ Cleaned English web text         │
│ GitHub                    │ 30 Billion   │ Open-source repos (MIT, Apache)  │
│ Books                     │ 26 Billion   │ Gutenberg + Book3 corpus         │
│ ArXiv                     │ 28 Billion   │ Scientific papers in LaTeX       │
│ Wikipedia                 │ 24 Billion   │ 20 languages                     │
│ StackExchange             │ 20 Billion   │ Question & Answer coding pairs   │
└───────────────────────────┴──────────────┴──────────────────────────────────┘
```

---

### B. SlimPajama: 627B Token Deduplicated Distillation (Cerebras, June 2023)

Although RedPajama-1T was a major milestone, it suffered from significant cross-source duplicate text, boilerplate repetition, and low-quality web snippets.

**Cerebras Systems** released **SlimPajama** (June 9, 2023):
* Developed a high-throughput, distributed **MinHashLSH (Locality-Sensitive Hashing)** pipeline.
* Extensively cleaned and deduplicated the entire 1.21T RedPajama corpus.
* **Removed 49.6% of raw bytes**, reducing the volume from **1,210B tokens down to 627B tokens**.

```
                REDUCING REDPAJAMA (1.21T) TO SLIMPAJAMA (627B)
┌─────────────────────────────────────────────────────────────────────────────┐
│  RedPajama-1T Raw Pool (1,210B Tokens / ~5.0 TB)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼ Distributed MinHashLSH Deduplication
                                       │ (13-gram shingles, 128 hash functions)
                                       │ Filters: Cross-slice duplicates,
                                       │ boilerplate lines, dead text
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SlimPajama (627B Tokens / ~2.5 TB)  ───► 49.6% Bytes Removed               │
│  • Common Crawl: 367B  • C4: 161B   • GitHub: 30B   • Books: 26B            │
│  • ArXiv: 28B          • Wiki: 24B  • StackExchange: 20B                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why SlimPajama Outperforms RedPajama:
1. **Higher Sample Efficiency:** Training on SlimPajama yields the same or better zero-shot downstream accuracy in half the training steps (and half the GPU compute budget).
2. **Reduced Memorization:** Repetitive sequences that cause catastrophic language generation loops are pruned.
3. **Better Upsampling Stability:** High-quality domains (Wikipedia, Books, ArXiv) are preserved intact while noisy web duplicates in Common Crawl are eliminated.

---

### C. RedPajama-V2 (Together AI, Late 2023)
Together AI expanded the vision with **RedPajama-V2**:
* **30+ Trillion tokens** across 84 Common Crawl dumps (2014–2023) covering English, German, French, Spanish, and Italian.
* Instead of releasing a single opinionated filtered cut, RedPajama-V2 computed **40+ quality signals per document**:
  * Statistical signals (line length, character entropy, punctuation density).
  * Model-based perplexity scores (Kneser-Ney, FastText, CCNet).
  * Toxicity, profanity, and duplication counts.
* Enables researchers to construct customized data mixtures tailored to specific compute and quality constraints.

---

## 6. FineWeb & FineWeb-Edu (Hugging Face, 2024)

Based on the landmark Hugging Face report (*"FineWeb: 15-trillion tokens, 44TB disk space dataset for LLM pretraining"* by Penedo et al., 2024), **FineWeb** and **FineWeb-Edu** represent the current state of the art in open web datasets.

```
                  ┌──────────────────────────────────────────────┐
                  │  96 Common Crawl Snapshots (2013 – 2024)     │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼ Heavy Rule-Based Cleaning & MinHash Deduplication
                  ┌──────────────────────────────────────────────┐
                  │  🍷 FineWeb (15 Trillion Tokens / 44 TB)      │
                  │  Permissive ODC-By 1.0 License               │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼ Educational Quality Classifier (Trained on Llama-3-70B)
                  ┌──────────────────────────────────────────────┐
                  │  📚 FineWeb-Edu                              │
                  ├──────────────────────────────────────────────┤
                  │  • Tier 1 (Very High Quality, Score ≥ 3):    │
                  │    1.3 Trillion Tokens (GPT-2 Tokenizer)     │
                  │  • Tier 2 (High Quality, Score ≥ 2):         │
                  │    5.4 Trillion Tokens (GPT-2 Tokenizer)     │
                  └──────────────────────────────────────────────┘
```

### A. FineWeb Core Characteristics
* **Volume & Disk Footprint:** **15 Trillion tokens** spanning **44 Terabytes** of disk space (uncompressed).
* **Data Origin:** Curated from **96 individual Common Crawl snapshots** from 2013 through 2024.
* **Licensing:** Released under the permissive **ODC-By 1.0** open data license.
* **Ablation Studies:** Hugging Face carefully ablated every stage:
  * Trafilatura-based HTML extraction.
  * Line and paragraph-level deduplication + global MinHashLSH.
  * Heuristic quality filters (C4, Gopher, language identification with FastText).

---

### B. FineWeb-Edu: Scalable Synthetic Educational Annotations
FineWeb-Edu is specifically optimized to teach LLMs high-density factual knowledge and scientific reasoning.

#### 1. The Annotation & Filtering Methodology:
1. **Seed Annotation via LLM-as-a-Judge:** 
   * A seed dataset of 500,000 web pages was scored by **`Meta-Llama-3-70B-Instruct`** on an educational scale from **0 to 5** (evaluating clarity, pedagogical value, depth of explanation, and factual density).
2. **Classifier Distillation:**
   * A lightweight, highly parallel classifier (`Snowflake-arctic-embed-m` + classification head) was trained on the Llama-3-70B annotations.
   * This classifier was executed across all 15 Trillion tokens of FineWeb.
3. **Filtering Tiers (measured with GPT-2 Tokenizer):**
   * **1.3 Trillion tokens (`score >= 3`):** Extremely high educational content density (used by Andrej Karpathy and modern GPT-2 reproductions).
   * **5.4 Trillion tokens (`score >= 2`):** Broad high-quality educational text.

#### 2. Benchmark Performance:
FineWeb-Edu significantly outperforms all prior open web datasets (RefinedWeb, RedPajama, SlimPajama, Dolma, C4) on educational benchmarks:
* **MMLU (Massive Multitask Language Understanding)**
* **ARC (AI2 Reasoning Challenge - Easy & Challenge)**
* **OpenBookQA & GSM8K**

---

### C. High-Throughput Binary Sharding Pipeline (`fineweb.py`)

In Andrej Karpathy's training setup, downloading raw JSON/Parquet files during training causes severe I/O and tokenization bottlenecks. Instead, FineWeb-Edu is pre-tokenized into compact binary shard files:

```python
# Streaming binary shard structure (100M tokens per shard):
# [Header: 256 uint32 ints (magic_number=20240520, version=1, num_tokens=100,000,000)]
# [Token IDs: raw uint16 array of size 100,000,000 elements]
```

```python
import numpy as np
import torch

def load_tokens_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch"
        assert header[1] == 1, "version mismatch"
        num_tokens = header[2]
        tokens = np.fromfile(f, dtype=np.uint16)
        assert len(tokens) == num_tokens
    return torch.tensor(tokens, dtype=torch.long)
```

---

## 7. Master Comparison Matrix

| Feature / Metric | GPT-2 (WebText) | GPT-3 Mixture | RedPajama-1T | SlimPajama (Cerebras) | FineWeb (Base) | FineWeb-Edu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Release Year** | 2019 | 2020 | 2023 | 2023 | 2024 | 2024 |
| **Organization** | OpenAI | OpenAI | Together AI | Cerebras | Hugging Face | Hugging Face |
| **Token Count** | ~10 Billion | ~300 Billion | ~1,210 Billion | ~627 Billion | **15 Trillion** | **1.3T / 5.4T** |
| **Primary Base** | Reddit Outbound Links | Common Crawl + 4 sources | LLaMA Recipe (7 domains) | Deduplicated RedPajama | 96 Common Crawl Dumps | FineWeb Filtered |
| **Quality Filter** | Reddit Karma ($\ge 3$) | Logistic Regression | FastText / CCNet | MinHashLSH + CCNet | C4/Gopher Heuristics | **Llama-3-70B Classifier** |
| **Deduplication** | Exact Match | MinHash (13-gram) | Per-slice Dedup | Global MinHashLSH (49.6% pruned) | Global MinHashLSH | Global MinHashLSH |
| **License / Openness** | Closed | Closed | Open (Apache-2.0) | Open (Apache-2.0) | Open (**ODC-By 1.0**) | Open (**ODC-By 1.0**) |
| **Downstream Strengths** | Basic fluency | In-context few-shot | General purpose | Compute-efficient LLaMA | Large web pretraining | **MMLU, ARC, Reasoning** |

---

## 8. Summary & Evolutionary Takeaways

1. **GPT-2's WebText (2019):** Human curation proxy (Reddit $\ge 3$ karma) yielded a clean 10B-token corpus.
2. **GPT-3 (2020):** Combined multi-source weighting with a machine-learned Logistic Regression filter on Common Crawl for 300B tokens.
3. **RedPajama-1T (2023):** Democratized the 7-domain LLaMA recipe into open source (~1.21T tokens).
4. **SlimPajama (2023):** Demonstrated that **extensive MinHashLSH deduplication** can remove ~50% redundant data to produce a superior 627B-token dataset.
5. **FineWeb & FineWeb-Edu (2024):** Set the new benchmark by scaling to 15T tokens and using **synthetic LLM annotations (Llama-3-70B)** to filter a 1.3T / 5.4T educational corpus.

