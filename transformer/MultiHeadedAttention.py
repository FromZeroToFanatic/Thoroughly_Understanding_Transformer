class MultiHeadedAttention(nn.Module):
    # 多头注意力机制
    def __init__(self, h, d_model, dropout=0.1):
        """
        h：表示注意力头的数量
        d_model：模型的维度，表示每个输入的向量大小
        dropout：dropout 的比率，默认值为 0.1
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # 创建了 4 个线性变换层，用来分别对查询、键、值进行投影，并生成最后的输出
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 确保同一个掩码应用到所有头
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) 实现了查询、键和值的线性投影
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) 调用前面定义的 attention 函数，计算多头注意力
        x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)
        # 3) 拼接 + 线性变换
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)