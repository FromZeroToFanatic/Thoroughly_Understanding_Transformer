class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder  # 编码器子模块
        self.decoder = decoder  # 解码器子模块
        self.src_embed = src_embed  # 源嵌入层
        self.tgt_embed = tgt_embed  # 目标嵌入层
        self.generator = generator  # 生成器模块(将解码器输出转换为词概率的生成器)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)  # 第一步：源序列编码
        output = self.decode(memory, src_mask, tgt, tgt_mask)  # 第二步：目标序列解码
        return output
        # return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        embedded_src = self.src_embed(src)  # 词嵌入 + 位置编码
        memory = self.encoder(embedded_src, src_mask)  # 编码器将嵌入和掩码作为输入
        return memory
        # return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        embedded_tgt = self.tgt_embed(tgt)  # 目标序列嵌入
        output = self.decoder(embedded_tgt, memory, src_mask, tgt_mask)
        return output
        # return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)