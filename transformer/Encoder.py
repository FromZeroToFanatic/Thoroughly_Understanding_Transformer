class Encoder(nn.Module):
    # Input Embedding → [EncoderLayer × N] → LayerNorm → Output
    # 每个 EncoderLayer 通常包含：多头自注意力、前馈网络、残差连接和层归一化
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        x：输入序列张量
        mask：掩码张量(用于控制注意力机制屏蔽掉填充位或非法位置)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)