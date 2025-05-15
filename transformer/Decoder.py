class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x：解码器的输入，一般是目标序列的嵌入
        memory：编码器的输出（Encoder 输出的表示）
        src_mask：源语言的掩码（通常用于屏蔽 padding）
        tgt_mask：目标语言的掩码（通常用于阻止看到后续词）
        """
        # 每一层会执行：自注意力子层(带因果 mask)、跨注意力子层(与 Encoder 输出交互)、前馈神经网络子层、残差连接和LayerNorm
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)