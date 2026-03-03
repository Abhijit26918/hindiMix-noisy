"""
Script 3: Generate synthetic ASR-style noise at 3 levels.

ASR Error Types (from literature):
  1. Character substitution (phonetically similar: v→w, sh→s, etc.)
  2. Word-level splits / merges  (dono → do no)
  3. Deletion of short function words (ne, ko, ka, the, is)
  4. Insertion of filler tokens (umm, uh, hmm)
  5. Transliteration variation (kya vs kiya vs kiyaa)
  6. Schwa deletion (Hindi: prakar → prkar)

Noise Levels:
  - Low:    ~1 error per 10 tokens
  - Medium: ~1 error per 5 tokens
  - High:   ~1 error per 3 tokens

Run: python scripts/noise_generation/03_add_noise.py
"""

import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

PROC_DIR = "data/processed"
NOISY_DIR = "data/noisy"
os.makedirs(f"{NOISY_DIR}/low", exist_ok=True)
os.makedirs(f"{NOISY_DIR}/medium", exist_ok=True)
os.makedirs(f"{NOISY_DIR}/high", exist_ok=True)

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# Phonetic substitution tables (Hindi-English code-mixed context)
# ─────────────────────────────────────────────────────────────────
CHAR_SUBS = {
    # English-side phonetic confusions
    "th": ["t", "d"],
    "ph": ["f", "p"],
    "sh": ["s", "ch"],
    "ch": ["sh", "c"],
    "v": ["w", "b"],
    "w": ["v", "u"],
    "z": ["j", "s"],
    "x": ["ks", "z"],
    "ck": ["k", "c"],
    "ee": ["i", "e"],
    "oo": ["u", "ou"],
    # Hindi transliteration confusions
    "aa": ["a", "ah"],
    "ii": ["i", "ee"],
    "uu": ["u", "oo"],
    "kh": ["k", "kha"],
    "gh": ["g", "gha"],
    "ng": ["n", "ngg"],
    "ya": ["ia", "a"],
    "wa": ["va", "a"],
}

HINDI_FUNCTION_WORDS = {"ne", "ko", "ka", "ki", "ke", "hai", "hain", "tha", "thi", "the", "se", "par", "mein", "me", "ho", "hoga"}
ENGLISH_FUNCTION_WORDS = {"the", "is", "are", "was", "a", "an", "of", "in", "on", "at", "to"}
FUNCTION_WORDS = HINDI_FUNCTION_WORDS | ENGLISH_FUNCTION_WORDS

FILLER_TOKENS = ["umm", "uh", "hmm", "err", "aah", "arre"]

TRANSLIT_VARIANTS = {
    "kya": ["kiya", "kyaa", "kia"],
    "nahi": ["nhi", "nahin", "naheen", "nai"],
    "bhi": ["bhe", "bi", "bii"],
    "hoga": ["hoga", "hoga", "hogaa"],
    "yaar": ["yar", "yaar", "yar"],
    "abhi": ["abhi", "abi", "abhii"],
    "acha": ["accha", "achaa", "acha"],
    "ek": ["ak", "ek", "eik"],
}


# ─────────────────────────────────────────────────────────────────
# Error injection functions
# ─────────────────────────────────────────────────────────────────

def char_substitution(word):
    """Replace a phonetically similar character sequence."""
    for src, tgts in CHAR_SUBS.items():
        if src in word.lower():
            replacement = random.choice(tgts)
            return word.lower().replace(src, replacement, 1)
    # fallback: random char swap
    if len(word) > 2:
        i = random.randint(0, len(word) - 1)
        chars = list(word)
        chars[i] = random.choice("aeiouklmnrst")
        return "".join(chars)
    return word


def word_split(word):
    """Split word into two parts (simulates incorrect ASR segmentation)."""
    if len(word) > 4:
        split_pt = random.randint(2, len(word) - 2)
        return word[:split_pt] + " " + word[split_pt:]
    return word


def delete_function_word(tokens):
    """Delete a random function word from the token list."""
    indices = [i for i, t in enumerate(tokens) if t.lower() in FUNCTION_WORDS]
    if indices:
        idx = random.choice(indices)
        tokens.pop(idx)
    return tokens


def insert_filler(tokens):
    """Insert a filler token at a random position."""
    pos = random.randint(0, len(tokens))
    tokens.insert(pos, random.choice(FILLER_TOKENS))
    return tokens


def transliteration_variant(word):
    """Replace with a common transliteration variant."""
    lower = word.lower()
    if lower in TRANSLIT_VARIANTS:
        return random.choice(TRANSLIT_VARIANTS[lower])
    return word


def schwa_deletion(word):
    """Simulate Hindi schwa deletion: remove 'a' from middle of word."""
    if len(word) > 4 and "a" in word[1:-1]:
        idx = word[1:-1].index("a") + 1
        return word[:idx] + word[idx + 1:]
    return word


# ─────────────────────────────────────────────────────────────────
# Apply noise to a single sentence
# ─────────────────────────────────────────────────────────────────

# Error rates: (errors per N tokens)
NOISE_RATES = {
    "low": 0.10,    # 10% of tokens get an error
    "medium": 0.20, # 20% of tokens
    "high": 0.35,   # 35% of tokens
}

ERROR_FUNCS = [char_substitution, word_split, transliteration_variant, schwa_deletion]


def add_noise(text: str, level: str) -> str:
    """Add ASR-style noise to a text at the given level."""
    rate = NOISE_RATES[level]
    tokens = text.split()
    if not tokens:
        return text

    new_tokens = []
    for token in tokens:
        if random.random() < rate:
            error_type = random.choices(
                ["char_sub", "split", "delete", "filler", "translit"],
                weights=[0.30, 0.15, 0.20, 0.10, 0.25],
            )[0]

            if error_type == "char_sub":
                new_tokens.append(char_substitution(token))
            elif error_type == "split":
                new_tokens.extend(word_split(token).split())
            elif error_type == "delete":
                if token.lower() not in FUNCTION_WORDS:
                    new_tokens.append(token)
                # else: skip (delete it)
            elif error_type == "filler":
                new_tokens.append(token)
                new_tokens.append(random.choice(FILLER_TOKENS))
            elif error_type == "translit":
                new_tokens.append(transliteration_variant(token))
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


# ─────────────────────────────────────────────────────────────────
# Main: generate all 3 noise levels
# ─────────────────────────────────────────────────────────────────

def generate_noisy_versions(input_file="data/processed/codemixed_clean.csv"):
    if not os.path.exists(input_file):
        input_file = "data/processed/merged_clean.csv"
    if not os.path.exists(input_file):
        print("[ERROR] No processed data found. Run scripts 01 and 02 first.")
        return

    df = pd.read_csv(input_file)
    print(f"[INFO] Loaded {len(df)} samples from {input_file}")

    # Generate 3x noisy versions per sample (each at different level)
    for level in ["low", "medium", "high"]:
        print(f"\n[INFO] Generating {level} noise...")
        noisy_texts = []
        for text in tqdm(df["text"].astype(str), desc=f"  {level}"):
            noisy_texts.append(add_noise(text, level))

        noisy_df = df.copy()
        noisy_df["text_original"] = df["text"]
        noisy_df["text"] = noisy_texts
        noisy_df["noise_level"] = level

        out_path = f"{NOISY_DIR}/{level}/noisy_{level}.csv"
        noisy_df.to_csv(out_path, index=False)
        print(f"  Saved {len(noisy_df)} samples → {out_path}")

    # Quick sanity check
    print("\n[INFO] Sample comparison (original vs noisy):")
    sample = df["text"].iloc[0]
    print(f"  Original: {sample}")
    for level in ["low", "medium", "high"]:
        print(f"  {level:8s}: {add_noise(sample, level)}")

    print("\n[INFO] Next: Run scripts/preprocessing/04_create_splits.py")


if __name__ == "__main__":
    generate_noisy_versions()
