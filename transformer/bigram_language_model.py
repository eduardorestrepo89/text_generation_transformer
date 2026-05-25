import torch
import torch.nn as nn
from torch.nn import functional as F 
device= 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):

    def __init__(self,n_embd,head_size,block_size,dropout):
        super().__init__()
        # Capas que generaran los querys keys y values, reciben el input que es la cantidad de embeddings representando cada token y retornan una cantidad de head size
        self.query=nn.Linear(n_embd,head_size, bias=False)
        self.key=nn.Linear(n_embd,head_size, bias=False)
        self.value=nn.Linear(n_embd,head_size, bias=False)
        self.dropout_reg=nn.Dropout(dropout)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    
    def forward(self, x):
        B,T,C=x.shape
        Q=self.query(x) # (B,T,head_size)
        K=self.key(x) # (B,T,head_size)

        # hacemos el primer poroducto punto de Q y K para obtener la matriz de afinidades entre los tokens
        # como debemos hacer coincidir las dimensiones en K tenemos que invertir o transponer las ultimas dos dimensiones, ya que necesitamos multiplicar una matris T*head_size por una head_size*T
        wei= Q @ K.transpose(-2,-1) * C**-0.5 #(B,T,head_size)@(B,head_size,T)->(B,T,T)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) 
        # finalmente normalizamos aplicando softmax
        wei=F.softmax(wei, dim=-1) 
        wei=self.dropout_reg(wei) #Vamos a apagar algunos de los pesos de las afinidades entre los tokens 
        v=self.value(x) 
        out=wei @ v #(B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_size,n_embd, block_size,dropout):
        super().__init__()
        self.heads=nn.ModuleList([Head(n_embd,head_size,block_size,dropout) for _ in range(num_heads)])
        self.projection=nn.Linear(n_embd,n_embd)
        self.dropout_reg=nn.Dropout(dropout)

    def forward(self,x):
        self_attention_output=torch.cat([head(x) for head in self.heads],-1) # se concatena a nivel de canal por eso el -1
        return self.dropout_reg(self.projection(self_attention_output))


class FeedForward(nn.Module):

    def __init__(self,n_embd,dropout):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd), #La multiplicacion por 4 viene del papper, no se explica mucho en el video 
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd), ##esta es la proyeccion 
            nn.Dropout(dropout),
        )  
    def forward(self,x):
        return self.net(x)
      

class Block(nn.Module):
    def __init__(self,n_embd, num_heads,block_size,dropout):
        super().__init__()
        head_size=n_embd//num_heads
        self.sa_heads=MultiHeadAttention(num_heads=num_heads,head_size=head_size,n_embd=n_embd,block_size=block_size,dropout=dropout) #karpathy manda como headsize el mismo num_embed y se divide por el numero de cabezas que se van a crear para que al final al concatenar tengamos el mismo n_embed que recibe la siguiente capa 
        self.ffwd=FeedForward(n_embd,dropout)
        self.layer_norm1=nn.LayerNorm(n_embd) # Pensemos en los layer norms como una normalizacion de las caracteristicas o inputs de un algoritmo de machine learning, donde queremos que las caracteristicas tengan desv estandar 1 y media 0, esto es lo que hace la layernorm pero a nivel de fila
        self.layer_norm2=nn.LayerNorm(n_embd)

    def forward(self,x):
        x= x + self.sa_heads(self.layer_norm1(x)) # aplicar la atencion o los head del mecanismo de atencion (B,T,C) 
        x= x + self.ffwd(self.layer_norm2(x)) # aplicar la computacion o el perceptron multicapa (B,T,C)
        return x

    
class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size,n_embd,block_size,num_heads, num_layers,dropout):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.posotion_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,num_heads,block_size,dropout) for _ in range(num_layers)])
        self.layer_norm=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
        self.block_size=block_size  

    def forward(self,idx,targets=None):
        B,T=idx.shape
        token_embd=self.token_embedding_table(idx) # Entra idx de tamaño B,T donde B es el batch size y T la cantidad de tokens de cada sample y retorna B,T,C donde C es el numero de canales o embedings representnado a cada token
        position_embd=self.posotion_embedding_table(torch.arange(T,device=device)) # (T,C) se obtiene un vector de tamaño T donde los valores son 0,1,2,3...T-1 o igual a block_size-1 para identificar los embedings de cada posicion
        
        #Lo que ocurre aqui es que pytorch toma la dimension de position_embeding y la transforma para que cuadre con la dimension de token_embedings
        # Asi, toma (T,C) -> (_,T,C) mira cual es la dimension a la izquierda en token_emb y transforma position embed a (B,T,C) para poder hacer la suma
        # entonces cada embeding de token tiene la suma del embeding de su posicion 
        x= token_embd+position_embd #(B,T,C)
        x= self.blocks(x)
        x=self.layer_norm(x)
        logits=self.lm_head(x) #(B,T,C) B es el batch size, T es la longitud de entrada del contexto y C es la cantidad de embedings de los tokens, lo que se obtiene en el retorno es una matriz B,T,Vocabsize donde la dimension vocab size representa los logits del modelo
        
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
            #Como tenemos embeddiongs de posicion, necesitamos que el context_example tenga la misma cantidad de tokens que el block size o lo que llamamos T, por lo cual lo recortaremos de atras para adelante 
            context_sample_croped=context_sample[:,-self.block_size:]
            logits, loss=self(context_sample_croped)
            logits=logits[: , -1 , :]
            probs=F.softmax(logits, dim=-1)
            context_sample_next=torch.multinomial(probs,num_samples=1)
            context_sample=torch.cat((context_sample,context_sample_next),dim=1)
        return context_sample