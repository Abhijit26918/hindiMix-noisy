"""
NoiseBridge: Phonetic-Weighted Noise-Invariant Contrastive Learning
for ASR-Robust Hate Speech Detection in Hindi-English Code-Mixed Text.

Architecture components:
  1. ByT5 encoder (byte-level, no OOV on noisy Hinglish)
  2. Noise-Aware Attention gate (learned token weighting)
  3. PWNIC loss — Phonetic-Weighted Noise-Invariant Contrastive
  4. Gradient Reversal Layer — adversarial noise disentanglement
  5. Auxiliary noise level predictor (multi-task)

Full loss (4 terms, each mathematically motivated):

  L_total = L_CE
          + α · L_PWNIC       (projection space: pushes clean≈noisy embeddings)
          - β · L_adv         (feature space: forces encoder to REMOVE noise info)
          + γ · L_aux         (multi-task: predicts noise level from noisy repr)

  Note the MINUS on L_adv: encoder MAXIMIZES adversarial loss (minimax game).
  This is representation disentanglement via gradient reversal.

  PWNIC operates on a separate projection head (not the classifier embedding)
  so contrastive and classification objectives do not compete in the same space.

PWNIC loss (novel):
  Standard InfoNCE weighted by phonetic dissimilarity φ ∈ [0,1]:
  L_PWNIC = -Σᵢ φᵢ · log[exp(sim(pᵢ_c,pᵢ_n)/τ) / Σₖ exp(sim(pᵢ_c,pₖ)/τ)]
  where φᵢ = 1 - JaroWinkler(x_clean_i, x_noisy_i)
  → pushes HARDER when noisy version sounds more different

Gradient Reversal (novel application):
  GRL(h, λ) = h                  (forward — identity)
  ∂GRL/∂h   = -λ · I             (backward — flip gradient)
  Noise predictor tries to classify noise level from h.
  GRL forces encoder to make h unpredictable w.r.t. noise level.
  Result: hate semantics preserved, noise characteristics removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import T5EncoderModel, AutoModel
import jellyfish


# ─────────────────────────────────────────────────────────────────
# Phonetic Utilities
# ─────────────────────────────────────────────────────────────────

def phonetic_similarity(text_a: str, text_b: str) -> float:
    """
    Word-level average Jaro-Winkler similarity.
    Returns [0, 1]. 1 = identical, 0 = completely different.
    """
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


def phonetic_weights(clean_texts: list, noisy_texts: list) -> torch.Tensor:
    """
    φ = 1 - PhoneticSimilarity ∈ [0, 1].
    High φ = sounds very different → needs more contrastive pressure.
    """
    weights = [1.0 - phonetic_similarity(c, n) for c, n in zip(clean_texts, noisy_texts)]
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# Gradient Reversal Layer
# ─────────────────────────────────────────────────────────────────

class GradientReversalFunction(Function):
    """
    Forward:  identity  →  GRL(h) = h
    Backward: flip sign →  ∂L/∂h_in = -λ · ∂L/∂h_out

    This makes any downstream predictor adversarial:
    the encoder learns to make h UNINFORMATIVE w.r.t. noise level.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Wraps GradientReversalFunction with a schedulable λ.

    λ is annealed from 0 → lambda_max during training using the
    standard DANN schedule: λ(p) = 2/(1+exp(-10p)) - 1
    where p = current_step / total_steps ∈ [0, 1]
    This prevents the adversarial signal from destabilizing early training.
    """

    def __init__(self, lambda_max: float = 1.0):
        super().__init__()
        self.lambda_max = lambda_max
        self.current_lambda = 0.0

    def set_lambda(self, p: float):
        """p = training progress ∈ [0, 1]"""
        self.current_lambda = self.lambda_max * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.current_lambda)


# ─────────────────────────────────────────────────────────────────
# PWNIC Loss
# ─────────────────────────────────────────────────────────────────

class PWNICLoss(nn.Module):
    """
    Phonetic-Weighted Noise-Invariant Contrastive Loss.

    For batch of N (clean, noisy) pairs:
      Positive: (z_clean_i, z_noisy_i)
      Negatives: all other 2(N-1) embeddings
      Weight φᵢ: phonetic dissimilarity of pair i

    L = -Σᵢ φᵢ · log[exp(sim(zᵢ_c,zᵢ_n)/τ) / Σₖ exp(sim(zᵢ_c,zₖ)/τ)]
    """

    def __init__(self, temperature: float = 0.15):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_clean: torch.Tensor, z_noisy: torch.Tensor,
                phi: torch.Tensor) -> torch.Tensor:
        N      = z_clean.size(0)
        device = z_clean.device

        z_clean = F.normalize(z_clean, dim=1)
        z_noisy = F.normalize(z_noisy, dim=1)
        z_all   = torch.cat([z_clean, z_noisy], dim=0)            # (2N, D)

        sim = torch.mm(z_all, z_all.T) / self.temperature         # (2N, 2N)
        sim.masked_fill_(torch.eye(2*N, dtype=torch.bool, device=device), float('-inf'))

        pos_idx = torch.cat([
            torch.arange(N, 2*N, device=device),
            torch.arange(0, N,   device=device),
        ])                                                         # (2N,)

        pos_sim    = sim[torch.arange(2*N, device=device), pos_idx]
        log_denom  = torch.logsumexp(sim, dim=1)
        loss_each  = -(pos_sim - log_denom)                        # (2N,)

        phi        = phi.to(device)
        phi_both   = torch.cat([phi, phi], dim=0)                  # (2N,)
        weight_sum = phi_both.sum().clamp(min=1e-8)

        return (phi_both * loss_each).sum() / weight_sum


# ─────────────────────────────────────────────────────────────────
# Noise-Aware Attention Gate
# ─────────────────────────────────────────────────────────────────

class NoiseAwareAttention(nn.Module):
    """
    Learned sigmoid gate per token.
    Implicitly learns to down-weight corrupted tokens.
    Gate output ∈ (0,1): low = noisy token, high = clean token.
    """

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
    Full NoiseBridge model.

    Loss terms:
      L_CE    — hate speech classification (main task)
      L_PWNIC — phonetic-weighted contrastive alignment (output space)
      L_adv   — adversarial noise disentanglement via GRL (feature space)
      L_aux   — noise level prediction, separate branch (multi-task)

    L_total = L_CE + α·L_PWNIC - β·L_adv + γ·L_aux

    Parameters:
      encoder_name : HuggingFace model (default: google/byt5-small)
      alpha        : PWNIC weight      (default: 0.15)
      beta         : adversarial weight (default: 0.3)
      gamma        : auxiliary weight  (default: 0.1)
      lambda_max   : GRL max reversal  (default: 1.0)
    """

    NOISE_LEVEL_MAP = {'clean': 0, 'low': 1, 'medium': 2, 'high': 3}

    def __init__(
        self,
        encoder_name:     str   = 'google/byt5-small',
        num_labels:       int   = 2,
        num_noise_levels: int   = 4,
        dropout:          float = 0.1,
        temperature:      float = 0.15,
        alpha:            float = 0.15,
        beta:             float = 0.3,
        gamma:            float = 0.1,
        lambda_max:       float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        # Encoder
        if 'byt5' in encoder_name.lower() or 't5' in encoder_name.lower():
            self.encoder     = T5EncoderModel.from_pretrained(encoder_name)
            self.hidden_size = self.encoder.config.d_model
        else:
            self.encoder     = AutoModel.from_pretrained(encoder_name)
            self.hidden_size = self.encoder.config.hidden_size

        # Noise-aware token gating
        self.noise_attention = NoiseAwareAttention(self.hidden_size)

        # Gradient Reversal Layer (sits between encoder output and noise predictor)
        self.grl = GradientReversalLayer(lambda_max=lambda_max)

        # Hate speech classifier
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        # Projection head for PWNIC contrastive loss only.
        # Separate from the classifier so contrastive and classification
        # objectives do not compete in the same representation space.
        # Following SimCLR: encoder z → proj_head → p (used only for PWNIC).
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Adversarial noise level predictor (receives GRL-reversed features)
        # Tries to predict noise level → encoder fights back via GRL
        self.adv_noise_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_noise_levels),
        )

        # Auxiliary noise level predictor (separate branch, NO reversal).
        # Applied to z_noisy so labels match what was actually corrupted.
        self.aux_noise_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_noise_levels),
        )

        # PWNIC loss
        self.pwnic = PWNICLoss(temperature=temperature)

    def set_grl_lambda(self, p: float):
        """Call each training step: p = step/total_steps ∈ [0,1]"""
        self.grl.set_lambda(p)

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode → noise-gated mean pool."""
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.noise_attention(out.last_hidden_state)
        mask   = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        input_ids:             torch.Tensor,
        attention_mask:        torch.Tensor,
        labels:                torch.Tensor = None,
        noisy_input_ids:       torch.Tensor = None,
        noisy_attention_mask:  torch.Tensor = None,
        phi:                   torch.Tensor = None,
        noise_level_labels:    torch.Tensor = None,
    ) -> dict:

        # ── 1. Encode clean ──────────────────────────────────────
        z_clean = self.encode(input_ids, attention_mask)

        # ── 2. Classification (hate / non-hate) ──────────────────
        logits  = self.classifier(self.dropout(z_clean))
        loss_ce = F.cross_entropy(logits, labels) if labels is not None else None

        total_loss  = loss_ce
        loss_pwnic  = None
        loss_adv    = None
        loss_aux    = None

        if noisy_input_ids is not None:
            # ── 3. Encode noisy ──────────────────────────────────
            z_noisy = self.encode(noisy_input_ids, noisy_attention_mask)

            # ── 4. PWNIC loss (projection space, not classifier space) ──
            # Project both embeddings before contrastive loss so the
            # classifier representation is not pulled by the contrastive obj.
            if phi is not None:
                p_clean    = self.proj_head(z_clean)
                p_noisy    = self.proj_head(z_noisy)
                loss_pwnic = self.pwnic(p_clean, p_noisy, phi)
                total_loss = total_loss + self.alpha * loss_pwnic

            # ── 5. Adversarial disentanglement via GRL ───────────
            # GRL reverses gradient → encoder removes noise info from z_noisy
            if noise_level_labels is not None:
                z_noisy_rev  = self.grl(z_noisy)
                adv_logits   = self.adv_noise_predictor(z_noisy_rev)
                loss_adv     = F.cross_entropy(adv_logits, noise_level_labels)
                # POSITIVE: predictor minimizes normally; GRL reverses encoder gradient
                total_loss   = total_loss + self.beta * loss_adv

            # ── 6. Auxiliary noise prediction (separate branch) ──
            # Applied to z_noisy — labels reflect what noise level was applied,
            # so z_noisy is the right input (clean text carries no noise signal).
            if noise_level_labels is not None:
                aux_logits = self.aux_noise_predictor(z_noisy)
                loss_aux   = F.cross_entropy(aux_logits, noise_level_labels)
                total_loss = total_loss + self.gamma * loss_aux

        return {
            'loss':       total_loss,
            'loss_ce':    loss_ce,
            'loss_pwnic': loss_pwnic,
            'loss_adv':   loss_adv,
            'loss_aux':   loss_aux,
            'logits':     logits,
        }
