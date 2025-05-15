class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        size: 表示输入向量的维度
        self_attn: 是一个自注意力机制模块
        feed_forward: 是前馈神经网络模块
        dropout: 是 dropout 比例
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 复制两份(自注意力层 + 前馈网络层)
        """
        x = LayerNorm(x)
        x = x + Dropout(SelfAttention(x))
        
        x = LayerNorm(x)
        x = x + Dropout(FeedForward(x))
        """
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 第一个子层：自注意力机制
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个子层：前馈神经网络
        return self.sublayer[1](x, self.feed_forward)