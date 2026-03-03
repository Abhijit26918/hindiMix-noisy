"""
NoiseRobust-HateDetect: Proposed Architecture

Key components:
  1. ByT5 encoder (byte-level → robust to character-level ASR errors)
  2. Phonetic feature layer (Soundex/Metaphone similarity)
  3. Noise-aware attention (down-weights likely-noisy tokens)
  4. Classification head

Why ByT5?
  - Operates at byte level → no OOV problem for noisy Hinglish
  - Naturally handles character substitutions, word splits
  - Proven effective for noisy text (Xue et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer
import jellyfish  # for phonetic similarity


class PhoneticFeatureExtractor(nn.Module):
    """
    Computes phonetic similarity features between original and likely-intended words.
    Used to signal to the model that certain tokens are likely ASR errors.
    """

    def __init__(self, feature_dim=64):
        super().__init__()
        self.proj = nn.Linear(4, feature_dim)  # 4 phonetic metrics → 64-dim

    def compute_phonetic_features(self, tokens: list[str]) -> torch.Tensor:
        """Returns a (len(tokens), 4) tensor of phonetic features."""
        features = []
        for token in tokens:
            soundex = jellyfish.soundex(token) if token.isalpha() else "0000"
            metaphone = jellyfish.metaphone(token) if token.isalpha() else ""
            jw = jellyfish.jaro_winkler_similarity(token, token)  # self-similarity = baseline

            # Features: soundex hash, metaphone length, word length, char diversity
            feat = [
                hash(soundex) % 1000 / 1000.0,        # soundex code (normalized)
                len(metaphone) / 10.0,                  # metaphone length
                len(token) / 20.0,                      # word length (normalized)
                len(set(token)) / max(len(token), 1),   # character diversity
            ]
            features.append(feat)
        return torch.tensor(features, dtype=torch.float32)

    def forward(self, tokens: list[str]) -> torch.Tensor:
        feats = self.compute_phonetic_features(tokens)  # (T, 4)
        return self.proj(feats)  # (T, feature_dim)


class NoiseAwareAttention(nn.Module):
    """
    Learns to down-weight tokens that appear to be ASR noise.
    A simple gating mechanism: gate = sigmoid(W * token_repr + b)
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        gates = torch.sigmoid(self.gate(hidden_states))  # (batch, seq_len, 1)
        return hidden_states * gates


class NoiseRobustHateDetector(nn.Module):
    """
    Full model: ByT5 + phonetic features + noise-aware attention + classifier.
    """

    def __init__(self, num_labels=2, phonetic_dim=64, dropout=0.1):
        super().__init__()

        # ByT5 encoder — byte-level, robust to character errors
        self.encoder = T5EncoderModel.from_pretrained("google/byt5-small")
        self.hidden_size = self.encoder.config.d_model  # 1472 for byt5-small

        # Phonetic feature projection
        self.phonetic_extractor = PhoneticFeatureExtractor(feature_dim=phonetic_dim)
        self.phonetic_proj = nn.Linear(phonetic_dim, self.hidden_size)

        # Noise-aware attention
        self.noise_attention = NoiseAwareAttention(self.hidden_size)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask, token_strings=None, labels=None):
        # 1. ByT5 encoding
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state  # (batch, seq_len, hidden_size)

        # 2. Add phonetic features (if token strings provided)
        if token_strings is not None:
            phon_feats = []
            for tokens in token_strings:
                feat = self.phonetic_extractor(tokens)  # (T, phonetic_dim)
                # Pad/truncate to seq_len
                seq_len = hidden.size(1)
                if feat.size(0) < seq_len:
                    pad = torch.zeros(seq_len - feat.size(0), feat.size(1))
                    feat = torch.cat([feat, pad], dim=0)
                else:
                    feat = feat[:seq_len]
                phon_feats.append(feat)
            phon_feats = torch.stack(phon_feats).to(hidden.device)  # (batch, seq_len, phonetic_dim)
            phon_proj = self.phonetic_proj(phon_feats)               # (batch, seq_len, hidden_size)
            hidden = hidden + phon_proj

        # 3. Noise-aware attention gating
        hidden = self.noise_attention(hidden)

        # 4. Pool (mean over non-padding tokens)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # 5. Classify
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# ─────────────────────────────────────────────────────────────────
# Quick model test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing NoiseRobustHateDetector...")
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = NoiseRobustHateDetector(num_labels=2)

    texts = ["yaar tu bahut bura hai", "i hate all those people"]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        labels=torch.tensor([1, 1]),
    )
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  Logits: {out['logits']}")
    print("  Model test passed!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
