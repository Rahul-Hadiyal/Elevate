# Environment Audit Report

**Date:** 2026-09-25  
**System:** Windows 11 (10.0.26200)  
**Workspace:** `C:\Users\sohan\Desktop\Hackathon`  

---

## 1. Hardware Specifications

| Component | Specification | Notes |
|---|---|---|
| **OS** | Windows 11 Pro 64-bit (10.0.26200-SP0) | Windows environment (PowerShell) |
| **CPU** | Intel64 (14 Physical cores, 20 Logical threads) | Highly capable for multi-threaded feature extraction & blocking |
| **System RAM** | 31.75 GB Total (16.08 GB Available) | Sufficient for in-memory sparse matrices and data structures |
| **Pagefile / Swap** | 2.00 GB | Monitor memory footprint during sparse matrix ops |
| **GPU / CUDA** | None (CUDA Available: `False`) | CPU-optimized computation (LightGBM, Scikit-learn, RapidFuzz) |

---

## 2. Python Runtime & Core Libraries

| Library / Tool | Version | License | Status | Purpose |
|---|---|---|---|---|
| **Python** | 3.13.7 (64-bit) | PSF | Verified | Core execution runtime |
| **LightGBM** | 4.7.0 | MIT | Verified | Primary Stage-1 and Stage-2 GBDT classifier |
| **XGBoost** | 3.4.1 | Apache 2.0 | Verified | Alternative/Ensemble tree model |
| **RapidFuzz** | 3.14.3 | MIT | Verified | Ultra-fast C++ string similarity & Levenshtein |
| **Scikit-Learn** | 1.9.0 | BSD-3-Clause | Verified | TF-IDF, Logistic Regression, Isotonic Regression |
| **Pandas** | 3.0.3 | BSD-3-Clause | Verified | TSV data loading and table processing |
| **NumPy** | 2.4.3 | BSD-3-Clause | Verified | Vectorized operations & Poisson-binomial DP |
| **SciPy** | 1.17.1 | BSD-3-Clause | Verified | Sparse matrices (CSR) & statistics |
| **PyYAML** | 6.0.3 | MIT | Verified | YAML configuration management |
| **Pytest** | 9.1.1 | MIT | Verified | Automated test suite execution |
| **PyTorch** | 2.12.1+cpu | BSD-style | Verified | CPU PyTorch runtime |

---

## 3. Constraints & Operational Policies

1. **No External Network Lookups:** Strict local offline processing. No external search, Google APIs, geocoders, or company registries.
2. **Model Licensing:** Final models must be MIT / Apache 2.0 compatible and $\le$ 8 Billion parameters (LightGBM, XGBoost fully compliant).
3. **No GPU Acceleration:** High multi-core CPU parallelism (20 threads) should be leveraged for candidate blocking and pairwise feature computation.
4. **Memory Management:** Use SciPy CSR sparse matrices for $n$-gram TF-IDF retrieval; avoid dense $N \times M$ cross-product allocations.
