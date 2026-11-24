#  Copyright (c) Prior Labs GmbH 2026.
"""
Hybrid multimodal fine-tuning wrapper for TabPFN v2.

Architecture:
  Structured modalities → TabPFN _embed_features (frozen, 内部含 z-score) → tokens
  Unstructured modalities → GatedMGM (LayerNorm) → CAP (trainable) → tokens
  All tokens + y token → TabPFN transformer blocks (LoRA) → cls_head (trainable)

This replaces the standard TabPFN forward() with a hybrid pipeline that calls
internal sub-modules directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_anything.model.lora import LoRALinear
from fusion_anything.model.projector import UnstructuredModalityProcessor


class TabPFNFineTuneWrapper(nn.Module):
    """Hybrid wrapper: TabPFN encoder + MGM+CAP projector + TabPFN transformer + cls_head.

    Structured features flow through TabPFN's frozen preprocessing pipeline
    (feature grouping → NaN/Inf handling → z-score → feature_group_embedder).
    Unstructured embeddings flow directly through the trainable GatedMGM → CAP
    projector, where GatedMGM applies per-sample LayerNorm before the bottleneck.
    All resulting tokens are concatenated with a target token and passed through
    TabPFN's transformer blocks (with LoRA adapters).  Classification uses a
    trainable head on top of the test-position target-token embeddings.
    """

    def __init__(
        self,
        tabpfn_model: nn.Module,
        unstructured_processor: UnstructuredModalityProcessor,
        max_n_classes: int = 10,
    ):
        super().__init__()
        self.tabpfn = tabpfn_model
        self.unstructured_processor = unstructured_processor
        self._max_n_classes = max_n_classes
        self.embed_dim = tabpfn_model.embedding_dim

        # Freeze all base TabPFN parameters; only LoRA + new modules are trainable
        for p in self.tabpfn.parameters():
            p.requires_grad_(False)
        # LoRA parameters are re-enabled below since they are nn.Parameters
        for m in self.tabpfn.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)

        # Modality-type embedding: one vector per modality index added to its tokens
        self.modality_emb = nn.Embedding(8, self.embed_dim)
        nn.init.normal_(self.modality_emb.weight, std=0.02)

        # Position embeddings for unstructured tokens (structured tokens receive
        # column embeddings from _embed_features internally)
        self.unstruct_pos_emb = nn.Embedding(512, self.embed_dim)
        nn.init.normal_(self.unstruct_pos_emb.weight, std=0.02)

        # Classification head — created lazily on first forward
        self.cls_head: nn.Sequential | None = None

    def _ensure_cls_head(self, device: torch.device) -> None:
        if self.cls_head is None:
            hidden = self.embed_dim * 4
            self.cls_head = nn.Sequential(
                nn.Linear(self.embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self._max_n_classes),
            ).to(device)

    def forward(
        self,
        modality_data: dict[str, torch.Tensor],
        y: torch.Tensor,
        train_pos: int,
        modality_names: list[str],
        modality_types: list[str],
        modality_dims: list[int],
        tabular_dim: int,
        dataset_K: int,
        n_classes: int,
    ) -> dict:
        """Hybrid forward pass.

        Args:
            modality_data: {name: tensor [B, N, d]} per modality.
            y: [B, N] integer labels. Context labels form target embeddings;
                Query labels are used to compute loss. Inference passes zeros
                at Query positions and discards the loss.
            train_pos: number of context rows; rows >= train_pos are queries.
            modality_names: ordered modality names.
            modality_types: 'structured' or 'unstructured' per modality.
            modality_dims: feature/embedding dimensionality per modality.
            tabular_dim: total structured feature dimension (metadata).
            dataset_K: Retained metadata parameter; this wrapper fixes CAP output
                to 24 tokens per unstructured modality.
            n_classes: number of classes for this episode.

        Returns:
            dict with keys: loss, cls_output, y_query, n_classes.
        """
        B, N = y.shape
        device = y.device
        self._ensure_cls_head(device)

        # ----------------------------------------------------------
        # Step 1: Structured features → TabPFN encoder (frozen)
        # ----------------------------------------------------------
        struct_parts: list[torch.Tensor] = []
        for name, mtype in zip(modality_names, modality_types):
            if mtype == "structured":
                struct_parts.append(modality_data[name])

        all_x_tokens: list[torch.Tensor] = []

        if struct_parts:
            x_struct = torch.cat(struct_parts, dim=-1)               # [B, N, d_struct]
            if self.training:
                perm = torch.randperm(x_struct.shape[-1], device=x_struct.device)
                x_struct = x_struct[:, :, perm]
            x_struct = x_struct.transpose(0, 1)                      # [N, B, d_struct]
            x_RSF, n_groups, n_rows, batch_size = (
                self.tabpfn._pad_and_group_features(x_struct)        # [N, B*G, 2]
            )
            struct_tokens, _ = self.tabpfn._embed_features(
                x_RSF,
                num_train_labels=train_pos,
                num_feature_groups=n_groups,
                batch_size=batch_size,
            )                                                        # [B, N, G, 192]
            all_x_tokens.append(struct_tokens)

        # ----------------------------------------------------------
        # Step 2: Unstructured embeddings → MGM+CAP (trainable)
        # ----------------------------------------------------------
        for name, mtype in zip(modality_names, modality_types):
            if mtype == "unstructured":
                x = modality_data[name]                              # [B, N, d_embed]
                tokens = self.unstructured_processor(
                    x, dataset_K=24,                                  # 非结构化模态固定 24 个 token
                )                                                    # [B, N, 24, 192]
                all_x_tokens.append(tokens)

        # ----------------------------------------------------------
        # Step 3: Concatenate x tokens + modality embeddings + column embeddings
        # ----------------------------------------------------------
        x_tokens = torch.cat(all_x_tokens, dim=2)                   # [B, N, G_total, 192]

        # Modality-type embedding: identifies which modality each token belongs to
        offset = 0
        mod_idx = 0
        if struct_parts:
            n = struct_tokens.shape[2]
            x_tokens[:, :, offset:offset + n] += self.modality_emb(
                torch.tensor(mod_idx, device=device),
            )
            offset += n
            mod_idx += 1
        for ut in all_x_tokens[1:] if struct_parts else all_x_tokens:
            n = ut.shape[2]
            x_tokens[:, :, offset:offset + n] += self.modality_emb(
                torch.tensor(mod_idx, device=device),
            )
            offset += n
            mod_idx += 1

        # Position embeddings for unstructured tokens.
        # Structured tokens already carry column embeddings from _embed_features.
        n_struct = struct_tokens.shape[2] if struct_parts else 0
        n_total = x_tokens.shape[2]
        n_unstruct = n_total - n_struct
        if n_unstruct > 0:
            pos_ids = torch.arange(n_unstruct, device=device)
            x_tokens[:, :, n_struct:] += self.unstruct_pos_emb(pos_ids)

        # ----------------------------------------------------------
        # Step 4: Target embedding (TabPFN, frozen)
        # ----------------------------------------------------------
        y_tabpfn = y[:, :train_pos].transpose(0, 1).float()         # [train_pos, B]
        embedded_y, _, _ = self.tabpfn._embed_targets(
            y_tabpfn, num_rows=N, num_train_labels=train_pos, batch_size=B,
        )                                                            # [B, N, 192]

        # ----------------------------------------------------------
        # Step 5: Concatenate features + target → transformer input
        # ----------------------------------------------------------
        all_tokens = torch.cat(
            [x_tokens, embedded_y.unsqueeze(2)], dim=2,
        )                                                            # [B, N, G_total+1, 192]

        # ----------------------------------------------------------
        # Step 6: TabPFN transformer blocks (LoRA-adapted, with gradient checkpointing)
        #
        # checkpoint wraps each block: forward activations are discarded after the
        # forward pass and recomputed during backward, reducing retained activation memory during gradient-enabled training.
        # We use a helper so the mutable container list is recreated on each call,
        # which is required because block.forward() pops from it.
        # ----------------------------------------------------------
        def _call_block(blk, x_tensor, eval_pos, save_peak):
            return blk([x_tensor], eval_pos, save_peak)[0]

        for block in self.tabpfn.blocks:
            all_tokens = torch.utils.checkpoint.checkpoint(
                _call_block, block, all_tokens, train_pos, None,
                use_reentrant=False,
            )

        # ----------------------------------------------------------
        # Step 7: Classification + loss
        # ----------------------------------------------------------
        test_y_tokens = all_tokens[:, train_pos:, -1]                # [B, N_query, 192]
        cls_output = self.cls_head(test_y_tokens)                    # [B, N_query, max_n_classes]
        cls_output = cls_output[:, :, :n_classes]                    # [B, N_query, n_classes]

        y_query = y[:, train_pos:].long()                            # [B, N_query]
        loss = F.cross_entropy(
            cls_output.reshape(-1, n_classes), y_query.reshape(-1),
            label_smoothing=0.1,
        )

        return {
            "cls_output_current": cls_output,
            "cls_output": cls_output,
            "loss": loss,
            "y_query": y_query,
            "n_classes": n_classes,
        }

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return all trainable parameters for the optimizer."""
        params: list[nn.Parameter] = []
        # Unstructured processor: GatedMGM + CAP
        params.extend(list(self.unstructured_processor.parameters()))
        # Classification head
        if self.cls_head is not None:
            params.extend(list(self.cls_head.parameters()))
        # Modality embeddings
        params.extend(list(self.modality_emb.parameters()))
        # Unstructured position embeddings
        params.extend(list(self.unstruct_pos_emb.parameters()))
        # LoRA parameters on TabPFN blocks
        for module in self.tabpfn.modules():
            from fusion_anything.model.lora import LoRALinear
            if isinstance(module, LoRALinear):
                params.extend([module.lora_A, module.lora_B])
        return params

    def count_trainable_params(self) -> int:
        """Count the parameters exposed by get_trainable_params()."""
        return sum(p.numel() for p in self.get_trainable_params())

    def add_new_params_to_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Register lazily-initialized parameters (e.g. cls_head) with the optimizer."""
        tracked = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                tracked.add(id(p))
        for p in self.get_trainable_params():
            if id(p) not in tracked:
                optimizer.add_param_group({"params": [p]})

    def clear_cuda_fragmentation(self) -> None:
        """Release unused cached CUDA allocations when CUDA is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
