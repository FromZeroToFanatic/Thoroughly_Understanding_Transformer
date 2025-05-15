class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        size: 表示输入的维度大小
        self_attn: 是一个自定义的自注意力模块
        src_attn: 是解码器中的跨注意力模块，表示与编码器输出的交互部分。
        feed_forward: 是一个前馈神经网络模块
        dropout: 是 dropout 概率，用于防止过拟合
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 复制三份（自注意力层 + 跨注意力层 + 前馈网络层）
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x: 解码器的输入，通常是目标序列的嵌入
        memory: 来自编码器的输出，用于跨注意力
        src_mask: 源语言的掩码
        tgt_mask: 目标语言的掩码
        """
        m = memory
        # 第一个子层：自注意力
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个子层：跨注意力
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 第三个子层：前馈神经网络
        return self.sublayer[2](x, self.feed_forward)