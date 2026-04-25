"""
NoiseBridge: Phonetic-Weighted Noise-Invariant Contrastive Learning
for ASR-Robust Hate Speech Detection in Hindi-English Code-Mixed Text.

Architecture:
  1. Encoder (ByT5 / mBERT / XLM-R / MuRIL)
  2. Noise-Aware Attention gate (learned token weighting)
  3. PWNIC contrastive loss (phonetic-weighted, projection space)
  4. Auxiliary noise-level predictor (multi-task, on z_noisy)

Per-row training scheme:

  Each batch row corresponds to exactly one example a baseline trainer
  would see — either a clean row or a noisy row. Every row carries:

    - clean_input_ids / noisy_input_ids   (one of the two is a placeholder
                                           when the row has no partner)
    - ce_mask_clean / ce_mask_noisy       (exactly one is True per row;
                                           selects which embedding gets CE)
    - has_pair                            (True iff both halves are real;
                                           gates PWNIC + aux)
    - phi, noise_level_labels             (used for paired rows only)

  Per-epoch CE update count = number of rows = N_clean + N_noisy_in_condition,
  exactly matching the baseline.

Loss:
  L_total = L_CE                            (per-row, anchor only)
          + alpha * L_PWNIC                 (paired rows only)
          + gamma * L_aux                   (paired rows only, if enable_aux)

Changes vs. v1 (the version that produced under-baseline numbers):

  1. REMOVED the GradientReversalLayer + adversarial noise predictor.
     The adversarial path and the auxiliary path were both attached to
     z_noisy with opposite gradient signs and were fighting each other.
     In per-noise-level runs the noise label was constant, which made the
     adversarial loss degenerate and pushed XLM-R high to 0.7773.

  2. CE supervision now matches the baseline exactly. v1 inner-merged
     clean and noisy and trained on the paired set only, where each pair
     contributed one CE update (on z_clean) — roughly half the supervision
     the baseline received. The per-row CE mask scheme above fixes this.

  3. `enable_aux` flag lets the trainer disable the aux head when the
     training set contains only one noise level, avoiding a constant
     classification target.

  4. Added a `predict()` method for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, AutoModel
import jellyfish


# ─────────────────────────────────────────────────────────────────
# Phonetic Utilities
# ─────────────────────────────────────────────────────────────────

def phonetic_similarity(text_a: str, text_b: str) -> float:
    """Word-level average Jaro-Winkler similarity. Returns [0, 1]."""
    words_a = str(text_a).lower().split()
    words_b = str(text_b).lower().split()
    if not words_a or not words_b:
        return 1.0
    n = max(len(words_a), len(words_b))
    scores = []
    for i in range(min(len(words_a), len(words_b))):
        try:
            scores.append(jellyfish.jaro_winkler_similarity(words_a[i], words_b[i]))
        except Exception:
            scores.append(1.0)
    scores += [0.0] * (n - len(scores))
    return sum(scores) / n


def phonetic_weights(clean_texts, noisy_texts) -> torch.Tensor:
    """phi = 1 - PhoneticSimilarity in [0, 1]. High phi = more contrastive pressure."""
    weights = [1.0 - phonetic_similarity(c, n) for c, n in zip(clean_texts, noisy_texts)]
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# PWNIC Loss
# ─────────────────────────────────────────────────────────────────

class PWNICLoss(nn.Module):
    """Phonetic-Weighted Noise-Invariant Contrastive Loss (InfoNCE style)."""

    def __init__(self, temperature: float = 0.15):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_clean: torch.Tensor, z_noisy: torch.Tensor,
                phi: torch.Tensor) -> torch.Tensor:
        N = z_clean.size(0)
        device = z_clean.device

        # Need at least 2 pairs for contrastive negatives.
        if N < 2:
            return torch.zeros((), device=device, requires_grad=True)

        z_clean = F.normalize(z_clean, dim=1)
        z_noisy = F.normalize(z_noisy, dim=1)
        z_all   = torch.cat([z_clean, z_noisy], dim=0)            # (2N, D)

        sim = torch.mm(z_all, z_all.T) / self.temperature         # (2N, 2N)
        sim.masked_fill_(torch.eye(2*N, dtype=torch.bool, device=device), float('-inf'))

        pos_idx = torch.cat([
            torch.arange(N, 2*N, device=device),
            torch.arange(0, N,   device=device),
        ])

        pos_sim   = sim[torch.arange(2*N, device=device), pos_idx]
        log_denom = torch.logsumexp(sim, dim=1)
        loss_each = -(pos_sim - log_denom)                        # (2N,)

        phi        = phi.to(device)
        phi_both   = torch.cat([phi, phi], dim=0)
        weight_sum = phi_both.sum().clamp(min=1e-8)

        return (phi_both * loss_each).sum() / weight_sum


# ─────────────────────────────────────────────────────────────────
# Noise-Aware Attention Gate
# ─────────────────────────────────────────────────────────────────

class NoiseAwareAttention(nn.Module):
    """Learned sigmoid gate per token; learns to down-weight corrupted tokens."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * torch.sigmoid(self.gate(hidden_states))


# ─────────────────────────────────────────────────────────────────
# NoiseBridge
# ─────────────────────────────────────────────────────────────────

class NoiseBridge(nn.Module):
    """
    NoiseBridge model — see module docstring for the per-row training scheme.
    """

    NOISE_LEVEL_MAP = {'clean': 0, 'low': 1, 'medium': 2, 'high': 3}

    def __init__(
        self,
        encoder_name: str   = 'google/byt5-small',
        num_labels:   int   = 2,
        num_noise_levels: int = 4,
        dropout:      float = 0.1,
        temperature:  float = 0.15,
        alpha:        float = 0.15,
        gamma:        float = 0.2,
        enable_aux:   bool  = True,
    ):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.enable_aux = enable_aux

        # Encoder
        if 'byt5' in encoder_name.lower() or 't5' in encoder_name.lower():
            self.encoder     = T5EncoderModel.from_pretrained(encoder_name)
            self.hidden_size = self.encoder.config.d_model
        else:
            self.encoder     = AutoModel.from_pretrained(encoder_name)
            self.hidden_size = self.encoder.config.hidden_size

        self.noise_attention = NoiseAwareAttention(self.hidden_size)

        # Hate speech classifier
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        # Projection head for PWNIC (separate from classifier so the
        # contrastive and classification objectives don't compete in the
        # same representation space).
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Auxiliary noise-level predictor (no GRL; standard multi-task head).
        self.aux_noise_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_noise_levels),
        )

        self.pwnic = PWNICLoss(temperature=temperature)

    # ── Encoding ─────────────────────────────────────────────────

    def encode(self, input_ids, attention_mask):
        """Encoder → noise-gated mean pool."""
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.noise_attention(out.last_hidden_state)
        mask   = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def predict(self, input_ids, attention_mask):
        """Inference path used at val/test time."""
        z = self.encode(input_ids, attention_mask)
        return self.classifier(self.dropout(z))

    # ── Training forward ─────────────────────────────────────────

    def forward(
        self,
        clean_input_ids,
        clean_attention_mask,
        noisy_input_ids,
        noisy_attention_mask,
        labels,
        ce_mask_clean,
        ce_mask_noisy,
        has_pair,
        phi,
        noise_level_labels,
        class_weights=None,
    ):
        device = clean_input_ids.device
        B      = labels.size(0)

        # ── Encode clean half (always) ──
        z_clean      = self.encode(clean_input_ids, clean_attention_mask)
        logits_clean = self.classifier(self.dropout(z_clean))

        ce_c_per = F.cross_entropy(
            logits_clean, labels, weight=class_weights, reduction='none'
        )
        ce_per_row = ce_c_per * ce_mask_clean.float()

        loss_pwnic = torch.zeros((), device=device)
        loss_aux   = torch.zeros((), device=device)

        # ── Encode noisy half if any row needs it ──
        # (either CE on noisy, or PWNIC/aux via a pair)
        need_noisy = bool((has_pair | ce_mask_noisy).any().item())

        if need_noisy:
            z_noisy      = self.encode(noisy_input_ids, noisy_attention_mask)
            logits_noisy = self.classifier(self.dropout(z_noisy))

            ce_n_per   = F.cross_entropy(
                logits_noisy, labels, weight=class_weights, reduction='none'
            )
            ce_per_row = ce_per_row + ce_n_per * ce_mask_noisy.float()

            # Paired rows: PWNIC + aux
            paired_idx = has_pair.nonzero(as_tuple=True)[0]

            # PWNIC needs ≥2 pairs for negatives.
            if paired_idx.numel() >= 2:
                p_clean    = self.proj_head(z_clean[paired_idx])
                p_noisy    = self.proj_head(z_noisy[paired_idx])
                phi_paired = phi[paired_idx]
                loss_pwnic = self.pwnic(p_clean, p_noisy, phi_paired)

            # Aux noise prediction on z_noisy[paired].
            if self.enable_aux and paired_idx.numel() > 0:
                aux_logits  = self.aux_noise_predictor(z_noisy[paired_idx])
                aux_targets = noise_level_labels[paired_idx]
                loss_aux    = F.cross_entropy(aux_logits, aux_targets)

        # Combined per-row CE — exactly one CE update per row, matching baseline.
        loss_ce    = ce_per_row.sum() / max(B, 1)
        total_loss = loss_ce + self.alpha * loss_pwnic + self.gamma * loss_aux

        return {
            'loss':       total_loss,
            'loss_ce':    loss_ce,
            'loss_pwnic': loss_pwnic,
            'loss_aux':   loss_aux,
            'logits':     logits_clean,
        }
