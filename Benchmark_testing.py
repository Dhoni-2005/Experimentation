
notebook_code = '''
# RTF-V4 Phase 1 — Baseline Benchmarking Notebook
# Platform: Kaggle (CPU-sufficient for this phase)
# Canonical Roadmap: Phase 1 – Environment Setup & Baseline Benchmarking
# Objective: Measure BPE fragmentation on Programming & Mathematics corpora
# before any RTF-V4 components are introduced.

# ============================================================
# CELL 1: INSTALLS & IMPORTS
# ============================================================
!pip install -q tiktoken tokenizers datasets transformers matplotlib seaborn

import os
import re
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# Standard BPE baselines
from transformers import AutoTokenizer
import tiktoken

# For dataset loading
from datasets import load_dataset

print("[RTF-V4 Phase 1] Environment ready.")

# ============================================================
# CELL 2: CONFIGURATION
# ============================================================
CONFIG = {
    "domains": ["programming", "mathematics"],
    "sample_size": 5000,           # Samples per domain for baseline
    "max_seq_length": 2048,        # Max tokens to process per sample
    "random_seed": 42,
    "output_dir": "/kaggle/working/rtf_v4_phase1_results",
    # Baseline tokenizers to compare
    "baseline_tokenizers": {
        "gpt2": "gpt2",            # HuggingFace GPT-2 BPE
        "gpt4": "cl100k_base",     # OpenAI GPT-4 tokenizer via tiktoken
    }
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
np.random.seed(CONFIG["random_seed"])

# ============================================================
# CELL 3: DATASET LOADING
# ============================================================
# Kaggle has built-in datasets; we use HuggingFace datasets for portability.
# Programming: The Stack (Python subset)
# Mathematics: ProofWiki-style math text + arXiv math abstracts

def load_domain_corpus(domain: str, n_samples: int = 5000):
    """Load curated corpus for a given domain."""
    texts = []
    
    if domain == "programming":
        # The Stack v2 - Python code
        print(f"[Data] Loading Programming corpus (The Stack - Python)...")
        try:
            ds = load_dataset("bigcode/the-stack-v2", "Python", split="train", streaming=True)
            for i, sample in enumerate(ds):
                if i >= n_samples:
                    break
                content = sample.get("content", "")
                if len(content) > 50:  # Filter out empty/trivial files
                    texts.append(content)
        except Exception as e:
            print(f"[Warning] HF load failed: {e}. Using synthetic fallback.")
            texts = generate_synthetic_programming(n_samples)
            
    elif domain == "mathematics":
        # arXiv math abstracts + synthetic proof-like text
        print(f"[Data] Loading Mathematics corpus (arXiv + synthetic)...")
        try:
            ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
            count = 0
            for sample in ds:
                abstract = sample.get("abstract", "")
                # Heuristic: keep math-heavy abstracts
                if any(k in abstract for k in ["theorem", "proof", "matrix", "equation", "∑", "∫", "∀", "∃", "ℝ", "ℕ"]):
                    texts.append(abstract)
                    count += 1
                if count >= n_samples // 2:
                    break
            # Fill remainder with synthetic math text
            texts += generate_synthetic_mathematics(n_samples - len(texts))
        except Exception as e:
            print(f"[Warning] HF load failed: {e}. Using synthetic fallback.")
            texts = generate_synthetic_mathematics(n_samples)
    
    print(f"[Data] {domain}: Loaded {len(texts)} samples.")
    return texts


def generate_synthetic_programming(n: int) -> list:
    """Generate synthetic Python-like code snippets for fallback."""
    snippets = []
    templates = [
        "def {func}(x, y):\\n    return x {op} y\\n",
        "class {cls}:\\n    def __init__(self, {arg}):\\n        self.{arg} = {arg}\\n",
        "for i in range({n}):\\n    print({func}(i))\\n",
        "import numpy as np\\narr = np.array([{nums}])\\nresult = np.dot(arr, arr.T)\\n",
        "if {cond}:\\n    {action}\\nelse:\\n    {alt}\\n",
    ]
    funcs = ["add", "subtract", "multiply", "compute", "transform", "normalize"]
    ops = ["+", "-", "*", "/", "//", "%", "**"]
    for _ in range(n):
        t = np.random.choice(templates)
        snippets.append(t.format(
            func=np.random.choice(funcs),
            op=np.random.choice(ops),
            cls=f"Class{np.random.randint(1000)}",
            arg=f"var{np.random.randint(100)}",
            n=np.random.randint(10, 1000),
            nums=", ".join(str(np.random.randint(0, 100)) for _ in range(5)),
            cond=f"x > {np.random.randint(100)}",
            action=f"return x * {np.random.randint(10)}",
            alt="return 0"
        ))
    return snippets


def generate_synthetic_mathematics(n: int) -> list:
    """Generate synthetic mathematical text for fallback."""
    snippets = []
    templates = [
        "Let f: ℝ → ℝ be defined by f(x) = {expr}. Then f is continuous on {domain}.",
        "Theorem: For all n ∈ ℕ, ∑_{{i=1}}^n i = n(n+1)/2.",
        "Consider the matrix A = [{matrix}]. The determinant is det(A) = {val}.",
        "Proof: Suppose ∀ε > 0, ∃δ > 0 such that |x - a| < δ implies |f(x) - L| < ε.",
        "The eigenvalues λ of the operator T satisfy the characteristic equation {eq} = 0.",
        "Integrate ∫_{{a}}^{{b}} {func} dx using substitution u = {sub}.",
    ]
    for _ in range(n):
        t = np.random.choice(templates)
        snippets.append(t.format(
            expr=f"x^{np.random.randint(2,5)} + {np.random.randint(1,10)}x + {np.random.randint(1,10)}",
            domain="ℝ" if np.random.random() > 0.5 else "[0, ∞)",
            matrix="; ".join(", ".join(str(np.random.randint(-10, 10)) for _ in range(3)) for _ in range(3)
            ),
            val=np.random.randint(-100, 100),
            eq=f"λ^{np.random.randint(2,4)} - {np.random.randint(1,10)}λ + {np.random.randint(1,10)}",
            func=f"sin({np.random.randint(1,5)}x)",
            sub=f"{np.random.randint(1,5)}x + {np.random.randint(1,10)}"
        ))
    return snippets


# Load corpora
CORPORA = {}
for domain in CONFIG["domains"]:
    CORPORA[domain] = load_domain_corpus(domain, CONFIG["sample_size"])

# ============================================================
# CELL 4: BASELINE TOKENIZER INITIALIZATION
# ============================================================
# We use GPT-2 (HF) and GPT-4 (tiktoken) as our universal BPE baselines.
# These represent the "standard" tokenizers that RTF-V4 seeks to improve upon.

TOKENIZERS = {}

# HuggingFace GPT-2 tokenizer
print("[Tokenizer] Loading GPT-2 BPE...")
TOKENIZERS["gpt2_hf"] = AutoTokenizer.from_pretrained("gpt2")

# OpenAI GPT-4 tokenizer
print("[Tokenizer] Loading GPT-4 cl100k_base...")
TOKENIZERS["gpt4_tiktoken"] = tiktoken.get_encoding("cl100k_base")

print("[Tokenizer] Baseline tokenizers ready.")

# ============================================================
# CELL 5: METRIC COMPUTATION
# ============================================================

def tokenize_hf(text: str, tokenizer) -> list:
    """Tokenize using HuggingFace tokenizer, return token strings."""
    return tokenizer.tokenize(text)


def tokenize_tiktoken(text: str, enc) -> list:
    """Tokenize using tiktoken, return decoded token strings."""
    token_ids = enc.encode(text, allowed_special="all")
    # tiktoken doesn't expose subword strings easily; we approximate
    # by decoding each ID individually to show fragmentation
    return [enc.decode([tid]) for tid in token_ids]


def compute_fertility(tokens: list, text: str) -> float:
    """
    Fertility rate = number of tokens / number of words.
    Lower is better (more compression).
    """
    words = text.split()
    if len(words) == 0:
        return 0.0
    return len(tokens) / len(words)


def compute_fragmentation_score(tokens: list, domain: str) -> dict:
    """
    Domain-specific fragmentation analysis.
    Returns percentage of terms that are split into subword fragments.
    """
    if domain == "programming":
        # Common Python keywords/constructs that should ideally be atomic
        atomic_terms = [
            "def", "class", "return", "import", "from", "if", "else", "for", 
            "while", "try", "except", "lambda", "self", "__init__", "numpy",
            "range", "print", "len", "append", "dict", "list", "tuple", "set"
        ]
    elif domain == "mathematics":
        # Math symbols and terms that should ideally be atomic
        atomic_terms = [
            "∑", "∫", "∀", "∃", "ℝ", "ℕ", "ℤ", "ℚ", "∈", "⊂", "∪", "∩",
            "theorem", "proof", "lemma", "corollary", "eigenvalue", "determinant",
            "matrix", "vector", "continuous", "differentiable", "integrable",
            "lim", "sin", "cos", "tan", "log", "exp", "sqrt", "pi", "infinity"
        ]
    else:
        atomic_terms = []
    
    fragmented = 0
    total_found = 0
    token_text = " ".join(tokens)
    
    for term in atomic_terms:
        if term in token_text:
            total_found += 1
            # Check if term appears as a single token or is split
            # A term is "fragmented" if no single token exactly matches it
            exact_match = any(t == term for t in tokens)
            if not exact_match:
                fragmented += 1
    
    frag_pct = (fragmented / total_found * 100) if total_found > 0 else 0.0
    return {
        "terms_checked": len(atomic_terms),
        "terms_found": total_found,
        "terms_fragmented": fragmented,
        "fragmentation_pct": frag_pct
    }


def compute_number_tokenization(tokens: list, text: str) -> dict:
    """
    Specific to H1: How are multi-digit numbers fragmented?
    Example: 8675309 → [867, 53, 09] or [8, 675, 309]
    """
    numbers = re.findall(r'\\b\\d{4,}\\b', text)  # Numbers with 4+ digits
    if not numbers:
        return {"total_numbers": 0, "avg_tokens_per_number": 0, "fully_atomic": 0}
    
    total_tokens_for_numbers = 0
    fully_atomic = 0
    
    for num_str in numbers:
        # Find tokens that compose this number
        # Simple heuristic: count tokens that are substrings of the number
        # and contiguous in the token stream
        num_tokens = 0
        for t in tokens:
            # Strip Ġ prefix (GPT-2 whitespace indicator)
            clean_t = t.replace("Ġ", "").replace("Ċ", "")
            if clean_t in num_str and len(clean_t) >= 1:
                num_tokens += 1
        total_tokens_for_numbers += max(num_tokens, 1)
        if num_tokens == 1:
            fully_atomic += 1
    
    return {
        "total_numbers": len(numbers),
        "avg_tokens_per_number": total_tokens_for_numbers / len(numbers),
        "fully_atomic": fully_atomic,
        "fully_atomic_pct": (fully_atomic / len(numbers) * 100)
    }


def benchmark_domain(domain: str, texts: list, tokenizer_name: str, tokenizer):
    """Run full benchmark suite on a domain corpus."""
    print(f"[Benchmark] Running {tokenizer_name} on {domain}...")
    
    results = {
        "domain": domain,
        "tokenizer": tokenizer_name,
        "n_samples": len(texts),
        "fertility_rates": [],
        "token_counts": [],
        "char_counts": [],
        "fragmentation": {"terms_checked": 0, "terms_found": 0, "terms_fragmented": 0, "fragmentation_pct": []},
        "number_tokenization": {"total_numbers": 0, "avg_tokens_per_number": [], "fully_atomic_pct": []},
        "timing": []
    }
    
    for text in texts:
        t0 = time.time()
        
        if tokenizer_name == "gpt4_tiktoken":
            tokens = tokenize_tiktoken(text, tokenizer)
        else:
            tokens = tokenize_hf(text, tokenizer)
        
        t1 = time.time()
        results["timing"].append(t1 - t0)
        
        # Basic metrics
        results["token_counts"].append(len(tokens))
        results["char_counts"].append(len(text))
        results["fertility_rates"].append(compute_fertility(tokens, text))
        
        # Fragmentation
        frag = compute_fragmentation_score(tokens, domain)
        results["fragmentation"]["terms_checked"] = frag["terms_checked"]
        results["fragmentation"]["terms_found"] += frag["terms_found"]
        results["fragmentation"]["terms_fragmented"] += frag["terms_fragmented"]
        if frag["terms_found"] > 0:
            results["fragmentation"]["fragmentation_pct"].append(frag["fragmentation_pct"])
        
        # Number tokenization
        num_tok = compute_number_tokenization(tokens, text)
        results["number_tokenization"]["total_numbers"] += num_tok["total_numbers"]
        if num_tok["total_numbers"] > 0:
            results["number_tokenization"]["avg_tokens_per_number"].append(num_tok["avg_tokens_per_number"])
            results["number_tokenization"]["fully_atomic_pct"].append(num_tok["fully_atomic_pct"])
    
    # Aggregate
    summary = {
        "domain": domain,
        "tokenizer": tokenizer_name,
        "n_samples": len(texts),
        "avg_tokens_per_sample": np.mean(results["token_counts"]),
        "avg_chars_per_sample": np.mean(results["char_counts"]),
        "avg_fertility": np.mean(results["fertility_rates"]),
        "median_fertility": np.median(results["fertility_rates"]),
        "std_fertility": np.std(results["fertility_rates"]),
        "total_terms_found": results["fragmentation"]["terms_found"],
        "total_terms_fragmented": results["fragmentation"]["terms_fragmented"],
        "overall_fragmentation_pct": (
            results["fragmentation"]["terms_fragmented"] / results["fragmentation"]["terms_found"] * 100
            if results["fragmentation"]["terms_found"] > 0 else 0
        ),
        "avg_tokens_per_number": np.mean(results["number_tokenization"]["avg_tokens_per_number"]) if results["number_tokenization"]["avg_tokens_per_number"] else 0,
        "avg_number_atomic_pct": np.mean(results["number_tokenization"]["fully_atomic_pct"]) if results["number_tokenization"]["fully_atomic_pct"] else 0,
        "avg_tokenization_time_ms": np.mean(results["timing"]) * 1000,
    }
    
    return summary, results


# ============================================================
# CELL 6: EXECUTE BASELINE BENCHMARKS
# ============================================================
all_summaries = []
all_raw_results = {}

for domain in CONFIG["domains"]:
    texts = CORPORA[domain]
    
    # GPT-2 HF
    summary_gpt2, raw_gpt2 = benchmark_domain(domain, texts, "gpt2_hf", TOKENIZERS["gpt2_hf"])
    all_summaries.append(summary_gpt2)
    all_raw_results[f"{domain}_gpt2"] = raw_gpt2
    
    # GPT-4 Tiktoken
    summary_gpt4, raw_gpt4 = benchmark_domain(domain, texts, "gpt4_tiktoken", TOKENIZERS["gpt4_tiktoken"])
    all_summaries.append(summary_gpt4)
    all_raw_results[f"{domain}_gpt4"] = raw_gpt4

# Convert to DataFrame
results_df = pd.DataFrame(all_summaries)
print("\\n" + "="*70)
print("RTF-V4 PHASE 1 — BASELINE BENCHMARK RESULTS")
print("="*70)
print(results_df.to_string(index=False))

# Save
results_path = os.path.join(CONFIG["output_dir"], "phase1_baseline_results.json")
results_df.to_json(results_path, orient="records", indent=2)
print(f"\\n[Saved] Results written to {results_path}")

# ============================================================
# CELL 7: VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RTF-V4 Phase 1: Baseline BPE Fragmentation Analysis", fontsize=14, fontweight="bold")

# 1. Fertility Rate by Domain & Tokenizer
ax = axes[0, 0]
sns.barplot(data=results_df, x="domain", y="avg_fertility", hue="tokenizer", ax=ax)
ax.set_title("Average Fertility Rate (Tokens / Words)")
ax.set_ylabel("Fertility")
ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Ideal (1.0)")
ax.legend()

# 2. Fragmentation Percentage
ax = axes[0, 1]
sns.barplot(data=results_df, x="domain", y="overall_fragmentation_pct", hue="tokenizer", ax=ax)
ax.set_title("Domain Term Fragmentation (%)")
ax.set_ylabel("Fragmented (%)")
ax.set_ylim(0, 100)

# 3. Number Tokenization (Avg tokens per number)
ax = axes[1, 0]
sns.barplot(data=results_df, x="domain", y="avg_tokens_per_number", hue="tokenizer", ax=ax)
ax.set_title("Multi-Digit Number Tokenization\\n(Avg Tokens per Number)")
ax.set_ylabel("Tokens per Number")
ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Ideal (1.0)")
ax.legend()

# 4. Sequence Length Inflation
ax = axes[1, 1]
results_df["chars_per_token"] = results_df["avg_chars_per_sample"] / results_df["avg_tokens_per_sample"]
sns.barplot(data=results_df, x="domain", y="chars_per_token", hue="tokenizer", ax=ax)
ax.set_title("Character Efficiency (Chars per Token)")
ax.set_ylabel("Characters / Token")

plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "phase1_baseline_plots.png")
plt.savefig(plot_path, dpi=150)
print(f"[Saved] Plots written to {plot_path}")
plt.show()

# ============================================================
# CELL 8: H1 NULL HYPOTHESIS STATEMENT
# ============================================================
print("\\n" + "="*70)
print("H1 NULL HYPOTHESIS VALIDATION")
print("="*70)
for _, row in results_df.iterrows():
    domain = row["domain"].upper()
    tok = row["tokenizer"]
    frag = row["overall_fragmentation_pct"]
    fert = row["avg_fertility"]
    num_tok = row["avg_tokens_per_number"]
    
    print(f"\\n{domain} | {tok}")
    print(f"  → Fertility: {fert:.3f} (higher = more tokens per word)")
    print(f"  → Fragmentation: {frag:.1f}% of domain terms are split")
    print(f"  → Number splitting: {num_tok:.2f} tokens per multi-digit number")
    
    if frag > 20 or num_tok > 1.5:
        print(f"  ⚠️  H1 SUPPORTED: Significant fragmentation detected.")
    else:
        print(f"  ✅ H1 NOT SUPPORTED: Fragmentation is low on this data.")

print("\\n" + "="*70)
print("PHASE 1 COMPLETE. Baseline established.")
print("Next: Phase 2 – Domain Dataset Preparation (high-purity curation)")
print("="*70)

# ============================================================
# CELL 9: PLACEHOLDER — FUTURE PHASE 5 (PERMANENT SEMANTIC REGISTRY)
# ============================================================
# INSTRUCTION: This cell is reserved for Phase 5 implementation.
# When the PSR is built, replace this block with:
#   - ROW_{DOMAIN}_### generation
#   - TAG assignment logic
#   - Registry lookup and mapping
#   - Re-run benchmarks H1-H5 against these identifiers
#
# Example structure:
#   PSR_ENTRY = {
#       "row_id": "ROW_PROGRAMMING_001",
#       "tags": ["TAG_FUNCTION", "TAG_LOOP"],
#       "domain": "programming",
#       "canonical_form": "for_loop",
#       "variants": ["for", "while", "foreach"]
#   }
# ============================================================
'''

# Save to output
output_path = "/mnt/agents/output/rtf_v4_phase1_kaggle_notebook.py"
with open(output_path, "w") as f:
    f.write(notebook_code)

print(f"Notebook saved to: {output_path}")
print(f"File size: {len(notebook_code)} characters")
