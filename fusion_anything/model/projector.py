"""
projector.py — 模态特征投影器

包含：
1. GatedMGM: 多头门控 MLP，LayerNorm → 共享瓶颈 → N head GLU → Dropout
2. CrossAttentionPooler (CAP): 双 Pre-LN + 双残差的交叉注意力池化器
3. SpatialPatchProjector: 空间特征图投影器（patch token → 192d）
4. UnstructuredModalityProcessor: GatedMGM + CAP + 可选空间投影

设计原则：
  - GatedMGM: LayerNorm(per-sample) → 共享瓶颈 → N=32 head GLU (value ⊙ sigmoid(gate)) → Dropout
  - LayerNorm 归一化沿特征维度，每样本独立，无需训练集统计量
  - max_dim 默认 1024；仅支持 d_in ≤ max_dim，较小维度在归一化后补零
  - GatedMGM 输出 32 token, CAP 池化为 ≤24 token (N > K)
  - CAP: Norm_q → Cross-Attn → +query残差 → Norm_ffn → FFN → +残差 → Norm_out
  - 空间路径：SpatialPatchProjector → CAP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== GatedMGM（多头门控 MLP）====================

class GatedMGM(nn.Module):
    """
    多头门控 MLP：将 d_in ≤ max_dim 的 flat embedding 映射为固定数量 token。

    结构:
      - LayerNorm 作为第一层，沿特征维度 per-sample 归一化
      - 共享瓶颈层实现全局维度交互（降维 + 非线性）
      - N=32 个独立 head，每 head 有独立 value/gate 权重
      - GLU 门控: token_i = value_i ⊙ sigmoid(gate_i)
      - Dropout(0.1) 防止 head 间 co-adaptation
      - 零填充支持任意 d_in ≤ max_dim
    """

    def __init__(
        self,
        max_dim: int = 1024,
        n_heads: int = 32,
        embed_dim: int = 192,
        bottleneck: int = 256,
    ):
        super().__init__()
        self.max_dim = max_dim
        self.n_heads = n_heads
        self.embed_dim = embed_dim

        # LayerNorm：沿特征维度 per-sample 归一化
        self.norm = nn.LayerNorm(max_dim)

        # 共享瓶颈：全局维度交互 + 非线性
        self.bottleneck = nn.Sequential(
            nn.Linear(max_dim, bottleneck, bias=False),
            nn.GELU(),
        )

        # N 个独立 head：每 head 有独立的 value/gate 权重 + bias
        self.W_v = nn.Parameter(torch.empty(n_heads, bottleneck, embed_dim))
        self.W_g = nn.Parameter(torch.empty(n_heads, bottleneck, embed_dim))
        self.b_v = nn.Parameter(torch.zeros(n_heads, embed_dim))
        self.b_g = nn.Parameter(torch.zeros(n_heads, embed_dim))
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_g)

        # Dropout：防止 32 head 之间的 co-adaptation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_in]  预提取的 flat embedding, d_in ≤ max_dim
        Returns:
            [B, N, n_heads, embed_dim]
        """
        d_in = x.shape[-1]
        if d_in > self.max_dim:
            raise ValueError(
                f"GatedMGM: input dim {d_in} exceeds max_dim {self.max_dim}"
            )

        # LayerNorm 在零填充之前，仅对真实维度做 per-sample 归一化
        x = F.layer_norm(
            x, (d_in,),
            weight=self.norm.weight[:d_in],
            bias=self.norm.bias[:d_in] if self.norm.bias is not None else None,
            eps=self.norm.eps,
        )

        if d_in < self.max_dim:
            x = F.pad(x, (0, self.max_dim - d_in))

        h = self.bottleneck(x)                                  # [B, N, bottleneck]
        value = torch.einsum("bnd,hde->bnhe", h, self.W_v) + self.b_v  # + 独立 bias
        gate = torch.sigmoid(torch.einsum("bnd,hde->bnhe", h, self.W_g) + self.b_g)
        return self.dropout(value * gate)


# ==================== CAP（交叉注意力池化器）====================

class CrossAttentionPooler(nn.Module):
    """
    CAP: Cross-Attention Pooler — 单块结构

    双 Pre-LN + 双残差的交叉注意力池化器：
      Learned Query → Norm_q → Cross-Attention(Q,K,V) → out_proj → +query(残差)
                                                                       ↓
                                      Norm_ffn → Linear→GELU→Linear → +残差
                                                                       ↓
                                                                   Norm_out
                                                                       ↓
                                                               truncate to target_K

    输入: x [B, N, n_tokens, embed_dim]  — GatedMGM 或 SpatialPatchProjector 输出的 token
    输出: x [B, N, K, embed_dim]         — K ≤ MAX_K, target_K 由数据集决定
    """

    MAX_K = 24

    def __init__(self, embed_dim: int = 192, n_heads: int = 8, max_k: int = None):
        super().__init__()
        if max_k is not None:
            self.MAX_K = max_k
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # 可学习查询向量
        self.query = nn.Parameter(torch.randn(1, 1, self.MAX_K, embed_dim))
        nn.init.normal_(self.query, std=0.02)

        # Pre-LN
        self.norm_q = nn.LayerNorm(embed_dim)

        # 多头交叉注意力
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # FFN (Pre-LN + residual)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, target_K: int = None) -> torch.Tensor:
        """Pool tokens into learned queries and optionally keep their leading subset.
        
        Args:
            x: Tensor shaped [batch, rows, tokens, embedding_dim].
            target_K: Output limit when smaller than the configured MAX_K.
        
        Returns:
            Tensor shaped [batch, rows, output_tokens, embedding_dim]."""
        B, N, n_tokens, E = x.shape

        x_flat = x.view(B * N, n_tokens, E)
        kv = self.kv_proj(x_flat)
        k, v = kv.chunk(2, dim=-1)

        query = self.query.expand(B, N, -1, -1).reshape(B * N, self.MAX_K, E)
        q_normed = self.norm_q(query)
        q = q_normed.view(B * N, self.MAX_K, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * N, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * N, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B * N, self.MAX_K, E)
        attn_out = self.out_proj(attn_out).view(B, N, self.MAX_K, E)

        h = query.view(B, N, self.MAX_K, E) + attn_out
        h = h + self.ffn(self.norm_ffn(h))
        h = self.norm_out(h)

        if target_K is not None and target_K < self.MAX_K:
            h = h[:, :, :target_K, :]

        return h


# ==================== 空间投影器 ====================

class SpatialPatchProjector(nn.Module):
    """空间 patch token 投影器：将图像编码器的空间特征图投影到模型 token 空间。

    输入: x [B, N, n_patches, d_patch]  (如 ResNet-18: n_patches=49, d_patch=512)
    输出: x [B, N, n_patches, embed_dim] (embed_dim=192)
    """

    def __init__(self, d_patch: int, embed_dim: int = 192):
        super().__init__()
        self.proj = nn.Linear(d_patch, embed_dim)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the last dimension while preserving batch, row, and patch axes."""
        return self.proj(x)


# ==================== 非结构化模态处理器 ====================

class UnstructuredModalityProcessor(nn.Module):
    """
    非结构化模态处理器：GatedMGM + CAP + 可选空间投影。

    处理路径:
      标准路径: flat embedding → GatedMGM (LayerNorm + 32 head GLU) → CAP (池化, ≤24 token)
      空间路径: patch tokens  → SpatialPatchProjector             → CAP (池化)

    GatedMGM 内部 LayerNorm 沿特征维度 per-sample 归一化后，
    通过 N=32 个独立 head + GLU 门控将 embedding 映射为 32 个差异化 token，
    CAP 使用可学习查询对 token 池化；该步骤不接收类别标签。
    """

    def __init__(
        self,
        embed_dim: int = 192,
        max_k: int = None,
        max_dim: int = 1024,
        n_heads: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mgm = GatedMGM(
            max_dim=max_dim,
            n_heads=n_heads,
            embed_dim=embed_dim,
        )
        self.cap = CrossAttentionPooler(
            embed_dim=embed_dim, n_heads=8, max_k=max_k,
        )

        self.spatial_proj = None
        self._spatial_d_patch = None

    def _ensure_spatial_proj(self, d_patch: int, device: torch.device):
        if self.spatial_proj is None or self._spatial_d_patch != d_patch:
            self.spatial_proj = SpatialPatchProjector(
                d_patch=d_patch, embed_dim=self.embed_dim
            ).to(device)
            self._spatial_d_patch = d_patch

    def forward(self, x: torch.Tensor, dataset_K: int = None) -> torch.Tensor:
        """Process flat features or spatial tokens, then apply attention pooling.
        
        Args:
            x: Flat features [batch, rows, width] or spatial patch features
                [batch, rows, patches, width].
            dataset_K: Optional output token limit passed to the attention pooler.
        
        Returns:
            Tokens shaped [batch, rows, output_tokens, embedding_dim]."""
        if x.ndim == 4 and x.shape[-1] != self.embed_dim:
            self._ensure_spatial_proj(x.shape[-1], x.device)
            x = self.spatial_proj(x)
        else:
            x = self.mgm(x)

        x = self.cap(x, target_K=dataset_K)
        return x
