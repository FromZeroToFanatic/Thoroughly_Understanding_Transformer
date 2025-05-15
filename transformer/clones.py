def clones(module, N):
    # 产生N个完全相同的网络层(深拷贝)
    """
    module：传入的一个网络层
    N：希望复制出的模块数量
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])