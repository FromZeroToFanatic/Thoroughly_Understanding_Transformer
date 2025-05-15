def attention(query, key, value, mask=None, dropout=None):
    # 缩放点积注意力
    """
    query、key 和 value：这三个参数是注意力机制中最核心的内容，分别代表查询、键和值，它们通常是由输入序列经过嵌入或上一层的计算得到的
    mask：掩码通常用于在计算注意力时忽略一些位置，尤其是在自注意力中避免模型看到未来的词（即在解码器中使用）
    dropout：为了防止过拟合，在注意力权重上应用 dropout
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 处理掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 计算注意力权重
    p_attn = F.softmax(scores, dim = -1)
    # Dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 计算加权值并返回结果
    return torch.matmul(p_attn, value), p_attn