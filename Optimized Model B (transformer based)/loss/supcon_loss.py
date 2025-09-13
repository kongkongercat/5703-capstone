import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Reference: Khosla et al. (https://arxiv.org/abs/2004.11362)

    Created by Meng Fanyi & Zhang Hang on 2025-09-11

    This implementation supports:
      - Camera-aware positive pairs (same ID, different camera)
      - LogSumExp trick for numerical stability
      - Handling samples that have no positive pairs in the batch

    Args:
        temperature (float): Temperature scaling factor (default: 0.07)

    Inputs:
        features: [B, D] feature vectors (before L2-normalization)
        labels:   [B] class IDs for each sample
        camids:   [B] (optional) camera IDs for each sample

    Output:
        loss: a scalar value
    """

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, camids=None):
        device = features.device
        B = features.shape[0]

        # Step 1: L2 normalize features
        z = F.normalize(features, dim=1)

        # Step 2: compute similarity matrix with temperature scaling
        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values  # numerical stability

        # Step 3: build positive mask based on labels (and camera if provided)
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)

        if camids is not None:
            camids = camids.view(-1, 1)
            # Only keep same-ID & different-camera pairs as positives
            pos_mask = pos_mask * (1.0 - torch.eq(camids, camids.T).float().to(device))

        # Step 4: remove diagonal (self-pairs)
        eye = torch.eye(B, device=device)
        logits_mask = 1.0 - eye
        pos_mask = pos_mask * logits_mask

        # Step 5: log-softmax over all other samples
        # use logsumexp for stability
        log_prob = logits - torch.logsumexp(
            logits + torch.log(logits_mask + 1e-12), dim=1, keepdim=True
        )

        # Step 6: only average over positive pairs
        pos_count = pos_mask.sum(1)  # number of positives per sample
        valid = pos_count > 0

        mean_log_prob_pos = torch.zeros_like(pos_count)
        mean_log_prob_pos[valid] = (pos_mask[valid] * log_prob[valid]).sum(1) / (
            pos_count[valid] + 1e-12
        )

        # Step 7: final loss (mean over valid samples)
        loss = -(mean_log_prob_pos[valid].mean() if valid.any() else log_prob.new_tensor(0.0))
        return loss
