class SublayerConnection(nn.Module):
    # 层归一化 + Dropout + 残差连接
    def __init__(self, size, dropout):
        """
        size: 表示输入向量的维度大小
        dropout: 表示 dropout 的概率
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        LayerNorm：让训练更稳定
        Dropout：正则化，防止过拟合
        残差连接：帮助梯度流动，缓解深层网络训练难的问题
        """
        return x + self.dropout(sublayer(self.norm(x)))