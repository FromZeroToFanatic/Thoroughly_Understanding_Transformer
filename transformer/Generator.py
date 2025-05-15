class Generator(nn.Module):
    # 定义生成器，由linear和softmax组成
    def __init__(self, d_model, vocab):
        """
        d_model：模型中每个时间步输出的隐藏向量的维度
        vocab：目标语言的词汇表大小
        """
        super(Generator, self).__init__()
        # nn.Linear是一个线性全连接层，作用是将模型输出的每个向量从大小 d_model 映射到 vocab 维度
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)