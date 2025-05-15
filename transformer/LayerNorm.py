class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        features：表示输入张量最后一个维度的大小
        eps：用于数值稳定性的一个小常数，防止除以0
        """
        super(LayerNorm, self).__init__()
        # a_2（对应公式中的γ）：初始化为 1 的可学习参数，用于缩放
        self.a_2 = nn.Parameter(torch.ones(features))
        # b_2（对应公式中的β）：初始化为 0 的可学习参数，用于偏移
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2