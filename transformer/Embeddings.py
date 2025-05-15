class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        d_model：表示词嵌入向量的维度，通常与模型的隐层维度一致
        vocab：词汇表的大小
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)