import torch
import torch.nn as nn
from torch.nn import functional as F 

class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        logits=self.token_embedding_table(idx) #(B,T,C) B es el batch size, T es la longitud de entrada del contexto y C es la cantidad de logits posibles o tokens posibles de mi vocabulario
        
        if targets is None:
            loss=None
        else: 
            B,T,C=logits.shape
            logits= logits.view(B*T,C)
            targets=targets.view(B*T)
            loss= F.cross_entropy(logits,targets)
        return logits, loss
    
    def generate(self, context_sample, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss=self(context_sample)
            logits=logits[: , -1 , :]
            probs=F.softmax(logits, dim=-1)
            context_sample_next=torch.multinomial(probs,num_samples=1)
            context_sample=torch.cat((context_sample,context_sample_next),dim=1)
        return context_sample