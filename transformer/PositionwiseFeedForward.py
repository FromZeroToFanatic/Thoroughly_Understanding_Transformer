class PositionwiseFeedForward(nn.Module):
    # 实现位置前馈神经网络 FNN
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model：模型的维度，输入和输出的向量大小
        d_ff：前馈网络中隐藏层的大小，通常比 d_model 大
        dropout：dropout 层的丢弃率，默认为 0.1，用来防止过拟合
        """
        super(PositionwiseFeedForward, self).__init__() \
        # 第一个线性层，将输入的维度从 d_model 映射到更大的隐藏层维度 d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 第二个线性层，将隐藏层的维度 d_ff 映射回 d_model，使得输出维度与输入维度相同
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))