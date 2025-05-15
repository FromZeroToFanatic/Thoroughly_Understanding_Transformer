def make_model(src_vocab, tgt_vocab, N=6,d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    src_vocab 和 tgt_vocab：源语言和目标语言的词汇表大小。源语言和目标语言的词汇表分别用于词嵌入层
    N=6：Transformer 的编码器和解码器中的层数（默认是 6 层）
    d_model=512：模型的隐藏层维度，也就是词嵌入的维度
    d_ff=2048：位置前馈网络的维度
    h=8：多头注意力机制中注意力头的数量
    dropout=0.1：在各个子层中使用的 dropout 比率，避免过拟合
    """
    c = copy.deepcopy
    # 创建多头注意力实例
    attn = MultiHeadedAttention(h, d_model)
    # 创建位置前馈网络实例
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 创建位置编码实例
    position = PositionalEncoding(d_model, dropout)
    # 创建 EncoderDecoder 模型
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    # 权重初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model