# Pretraining Datasets Guide: WebText (GPT-2) vs. GPT-3 Data Mixture

A comprehensive architectural and engineering guide on the pretraining datasets used in **GPT-2 (WebText)** and **GPT-3 (Common Crawl Mixture & WebText2)**, their curation pipelines, filtering mechanisms, deduplication techniques, and modern successors (e.g., **FineWeb-Edu**).

---

## 1. Overview & Evolution of Pretraining Corpora

Language modeling transitioned from small, human-annotated academic benchmarks (e.g., Penn Treebank, WikiText-103) to web-scale uncurated and heuristically filtered corpora. 

| Model | Dataset | Raw Volume | Training Tokens | Key Philosophy |
|:---|:---|:---:|:---:|:---|
| **GPT-1 (2018)** | BookCorpus | ~4.5 GB | ~1 Billion | Long contiguous spans of structured narrative text. |
| **GPT-2 (2019)** | **WebText** | ~40 GB (8M docs) | ~10 Billion | Human-curated web links via Reddit karma thresholding. |
| **GPT-3 (2020)** | **Filtered Common Crawl + WebText2 + Books + Wiki** | ~570 GB (post-filter) | ~300 Billion | Multi-source weighted blend, classifier-based quality filtering, MinHash deduplication. |
| **Modern (2024+)** | **FineWeb / FineWeb-Edu / RedPajama-V2** | 15+ Trillion tokens | 1T – 15T+ | Advanced heuristic classifiers + LLM-as-a-judge quality scoring. |

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
   - The community created open-source reproductions:
     - **OpenWebText (2019):** Scrapes Reddit submissions through 2018 with $\ge 3$ karma (~38 GB text, ~9B tokens).
     - **OpenWebText2 (EleutherAI, 2020):** Extended Reddit scraping to 2020, added cleaner deduplication and content filters for The Pile.

---

## 3. GPT-3 Training Dataset: The 300B-Token Mixture

### Why WebText Was Insufficient for GPT-3
For the 175B-parameter GPT-3 model, training for 300 billion tokens on the 10B-token WebText would have meant repeating the same data 30 times, causing catastrophic overfitting and performance plateaus. OpenAI therefore expanded to web-scale **Common Crawl**, but introduced an extensive 3-stage cleaning pipeline.

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

To make Common Crawl usable, OpenAI developed a 3-part filtering and curation pipeline:

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

### 1. Classifier-Based Quality Filtering
* Trained a **fast Logistic Regression text classifier** using word-level and sentence-level features.
* **Positive Training Set:** High-quality trusted text (WebText, Books, Wikipedia).
* **Negative Training Set:** Raw, unfiltered Common Crawl web pages.
* Each document was scored; documents with low quality scores were probabilistically discarded.

### 2. MinHash Deduplication
* High levels of duplicate text harm generalization and increase memorization.
* Document-level deduplication used **MinHashLSH with 13-grams** to detect and remove near-duplicate documents within and across crawl dumps.

### 3. Contamination Filtering
* For zero-shot and few-shot evaluation integrity, OpenAI ran n-gram overlap checks against test sets of downstream benchmarks (LAMBADA, Winograd, PIQA, ARC, OpenBookQA).
* When test overlap was identified, contaminated benchmark samples were flagged and reported.

---

## 5. Direct Comparison: WebText (GPT-2) vs. WebText2 vs. GPT-3 Mixture

| Dimension | GPT-2 (WebText) | GPT-3 (WebText2) | GPT-3 Full Mixture |
|:---|:---|:---|:---|
| **Release Year** | 2019 | 2020 | 2020 |
| **Data Sources** | Outbound Reddit links ($\ge 3$ karma) | Outbound Reddit links ($\ge 3$ karma) from newer time period | 5 distinct sources (Filtered CC, WebText2, Books1, Books2, Wikipedia) |
| **Raw Text Volume** | 40 GB | ~90 GB | > 1 TB raw before filtering |
| **Token Count** | ~8–10 Billion | ~19 Billion | ~300 Billion sampled (from ~500B pool) |
| **Quality Filter Mechanism** | Implicit (human Reddit karma threshold $\ge 3$) | Implicit (Reddit karma threshold) + updated text extraction | Supervised Logistic Regression Classifier + MinHashLSH |
| **Deduplication** | Exact / heuristic doc deduplication | Exact & MinHash deduplication | Document-level MinHashLSH across 45TB crawl |
| **Wikipedia Inclusion** | ❌ **Excluded** (avoid benchmark contamination) | ❌ **Excluded** | ✅ **Included** (3% mixture weight, ~3.4 epochs) |
| **Books Inclusion** | ❌ **No dedicated books corpus** | ❌ **No dedicated books corpus** | ✅ **Included** (Books1: 8%, Books2: 8%) |
| **Multi-Source Sampling** | Uniform single-source | Uniform single-source | Temperature-weighted multinomial sampling per source |

---

## 6. Modern Datasets & Andrej Karpathy's Implementation (FineWeb-Edu)

In modern GPT-2 reproductions (such as Andrej Karpathy's *build-nanogpt* video):
* Instead of scraping Reddit or filtering raw Common Crawl from scratch, researchers use modern pre-curated open datasets like **Hugging Face FineWeb-Edu**.

### What is FineWeb-Edu?
* **FineWeb-Edu** is a 1.3T token (or 10B/100B subset) extracted from 96 Common Crawl dumps (2013–2024).
* **Classifier:** Scored using **Llama-3-70B-Instruct** as an educational quality judge, which trained an ultra-fast synthetic classifier.
* Provides significantly higher quality tokens per byte than original 2019 WebText or 2020 Common Crawl, reaching GPT-2 performance with fewer total training steps.

### Binary Sharding Pipeline (`fineweb.py`)
To train high-throughput models without Python string/tokenization bottleneck during the loop:
1. Tokenize text into NumPy arrays (`dtype=np.uint16` for vocabulary $\le 65,536$).
2. Write tokens into binary shards (`.bin` files of 100M tokens each).
3. The data loader (`DataLoaderLite`) loads shards directly using memory mapping (`np.memmap`) or sequential buffer reading.

```python
# Streaming binary shard structure (100M tokens per shard)
# [Header: 256 ints (magic number, version, num_tokens)] [Token IDs: uint16 array ...]
```

---

## 7. Summary & Takeaways

1. **GPT-2's WebText** proved that high-quality human curation (Reddit karma filter) alone could train a coherent language model on 40 GB of text without task-specific labels.
2. **GPT-3's Data Recipe** demonstrated that scaling to 300B+ tokens requires combining **massive filtered web crawls (Common Crawl)** with **oversampled high-quality sources (WebText2, Wikipedia, Books)** using machine-learning-based quality classifiers and MinHash deduplication.
3. Modern open-source pretraining leverages **FineWeb-Edu**, yielding higher sample efficiency and zero-shot performance for reproduction workflows.
