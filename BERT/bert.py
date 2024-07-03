import torch
import torch.nn as nn


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, num_segments, max_len, embed_dim, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.segmt_embed = nn.Embedding(num_segments, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)], )
    
    def forward(self, sequence, segment):
        embed_val = self.token_embed(sequence) + self.segmt_embed(segment) + self.pos_embed(self.pos_inp)
        return embed_val
    
    
class BERT(nn.Module):
    def __init__(self, vocab_size, num_segments, max_len, embed_dim, num_layers, attention_heads, dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, num_segments, max_len, embed_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, attention_heads, embed_dim * 4)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, num_layers)
        
    def forward(self, sequence, segment):
        out = self.embedding(sequence, segment)
        out = self.encoder_block(out)
        return out
        

if __name__ == "__main__":
    vocab_size = 30000
    num_segments = 3
    max_len = 512
    embed_dim = 768
    num_layers = 12
    attention_heads = 12
    dropout = 0.1

    sample_sequence = torch.randint(high = vocab_size, size = [max_len, ])
    sample_segment = torch.randint(high = num_segments, size = [max_len, ])

    embedding = BERTEmbedding(vocab_size, num_segments, max_len, embed_dim, dropout)
    embedding_tensor = embedding(sample_sequence, sample_segment)
    print(embedding_tensor.size())
    
    bert = BERT(vocab_size, num_segments, max_len, embed_dim, num_layers, attention_heads, dropout)
    out = bert(sample_sequence, sample_segment)
    print(out.size())