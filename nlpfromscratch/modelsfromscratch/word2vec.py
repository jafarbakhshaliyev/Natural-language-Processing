import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy


class CBOW(nn.Module):
  def __init__(self, embedding_size = 100, vocab_size = -1, window_size = 2):
    super(CBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_size)
    self.linear = nn.Sequential(
        nn.Linear(embedding_size, 128),
        nn.ReLU(),
        nn.Linear(128, vocab_size)
    )
    self.window_size = window_size

  def forward(self, inputs):
    embeddings = torch.sum(self.embeddings(inputs), dim=1)
    return self.linear(embeddings)

class TextDataset(Dataset):
    def __init__(self, data_file_path, window_size=2):
        with open(data_file_path,'r',encoding='utf-8') as f:
            s = f.read().lower()
        
        nlp = spacy.load('en_core_web_sm')   
        words_tokenized = [token.text for token in nlp(s)]  

        self.context_target =  [([words_tokenized[i-(j+1)] for j in range(window_size)] +\
                                 [words_tokenized[i+(j+1)] for j in range(window_size)],
                                words_tokenized[i])
                                for i in range(window_size, len(words_tokenized)-window_size)]

        self.vocab = Counter(words_tokenized)
        self.word_to_idx = {word_tuple[0]: idx for idx, word_tuple in enumerate(self.vocab.most_common())}
        self.idx_to_word = list(self.word_to_idx.keys())
        self.vocab_size = len(self.vocab)
        self.window_size = window_size

    def __getitem__(self, idx):
        context = torch.tensor([self.word_to_idx[w] for w in self.context_target[idx][0]])
        target = torch.tensor([self.word_to_idx[self.context_target[idx][1]]])
        return context, target

    def __len__(self):
        return len(self.context_target)
    
    