q = [B, T, H, K] -> [B, H, T, K]
k = [B, T, H, K] -> [B, H, T, K]
g = [B, T, H, K] -> [B, H, T, K]
chunk_size = 64
BT = chunk_size
BK = 128
NT = cdiv(T, BT)
按照公式来看 A = \underbrace{\text{scale} \cdot \mathbf{q}_t^\top (\mathbf{\Lambda}_{j \to t} \odot \mathbf{k}_j) }_{\text{Part 2: Intra-chunk (块内贡献)}}, 我们直接计算
q_blockshape = [BT, BK]
k_blockshape = [BT, BK]
g_blockshape = [BT, BK]
qk = jnp.dot(q, k.transpose(0, 1)) # [BT, BT]

A *= scale
A = A.masked_fill(~jnp.tril(jnp.zeros(BT, BT)), 0)