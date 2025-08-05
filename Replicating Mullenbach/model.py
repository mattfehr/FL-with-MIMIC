from gensim.models import Word2Vec
import torch.nn            as nn
import torch.nn.functional as F
from math import floor
import torch

class ConvAttnPool(nn.Module):
    def __init__(self, label_space = 50, embed_table = None,vocab_size = None, embed_d = None, num_of_filters = 10, kernel_size = 3, drop_out = 0.2):
        super().__init__()
        #
        model = Word2Vec.load('/home/drew/FL-with-MIMIC/Replicating Mullenbach/processed_full.w2v')
        vocab_size, embed_d = model.wv.vectors.shape
        # print(f'{vocab_size=}', f'{embed_size=}')
        embed_table = torch.from_numpy(model.wv.vectors).type(torch.float32)
        embed_table = torch.concat([embed_table,torch.zeros(size = (1,embed_d))], dim = 0)

        #
        #self.embed = nn.Embedding(num_embeddings = num_of_words ,embedding_dim=embed_d, padding_idx=0)
        self.embed  = nn.Embedding.from_pretrained(embeddings=embed_table,padding_idx=vocab_size)
        self.conv  = nn.Conv1d(embed_d, num_of_filters, kernel_size = kernel_size, padding = int(floor(kernel_size//2)))
        self.U     = nn.Linear(num_of_filters, label_space)
        self.final = nn.Linear(num_of_filters, label_space)
        self.embed_drop = nn.Dropout(p = drop_out)
        self.embedding_size = embed_d
        #self.embed.requires_grad_ = False

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose (1, 2)
        x = F. tanh(self.conv(x).transpose(1,2))
        alpha = F.softmax(self.U.weight.matmul(x.transpose (1,2)) , dim=2)
        m     = alpha.matmul(x)
        y     = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y, alpha
    

def GenerateModel(num_of_filters = 15,kernel_size = 5):
    # This requires the other .w2v files as well.
    model = Word2Vec.load('processed_full.w2v')
    vocab_size, embed_size = model.wv.vectors.shape
    # print(f'{vocab_size=}', f'{embed_size=}')
    embedding_table = torch.from_numpy(model.wv.vectors).type(torch.float32)
    embedding_table = torch.concat([embedding_table,torch.zeros(size = (1,embed_size))], dim = 0)
    # print(f'{vocab_size=}', f'{embed_size=}') # Keep the since vocab_size refers to the last index, our padding index)

    return ConvAttnPool(
            drop_out       = 0.2,
            embed_table    = embedding_table,
            vocab_size     = vocab_size,
            num_of_filters = num_of_filters, # Filters in paper -> 10
            label_space    = 50, 
            kernel_size    = kernel_size,
            embed_d        = embed_size)