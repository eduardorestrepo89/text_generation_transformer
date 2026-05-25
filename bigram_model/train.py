import torch
import torch.nn as nn
from bigram_language_model import BigramLanguageModel
torch.manual_seed(1337)

# Hyperparameters
device= 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size=32 #numero de ejemplos independientes del texto que voy a tomar
block_size=8 #tamaño de la secuencia de tokens que se usaran para predecir, si la secuencia de entrada es mayor toca recortar
epochs=10000
eval_iters=200
eval_interval=1000
lr=1e-3
#----------

# Read the text

with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars=sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode= lambda str: [stoi[ch] for ch in str]
decode= lambda integers: "".join([itos[integer] for integer in integers])

data= torch.tensor(encode(text), dtype=torch.long)

# Train validation split

train_size = int(0.90*len(data))
train_data= data[:train_size]
val_data= data[train_size:]

def get_batch(split):
    data= train_data if split == 'train' else val_data
    ix= torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device), y.to(device)
    return x,y

model=BigramLanguageModel(vocab_size)
model=model.to(device)

@torch.no_grad() #para que no calcule los gradientes y los almacene, como no se va a hacer un backward no es necesario calcular gradientes
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for eval_iter in range(eval_iters):
            xb,yb=get_batch(split)
            logits,loss=model(xb,yb)
            losses[eval_iter]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out
    
optimizer=torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    
    if epoch % eval_interval == 0:
        model_metrics=estimate_loss()
        print(f"Epoch {epoch}: train loss {model_metrics['train']} val loss {model_metrics['val']}")

    xb,yb=get_batch('train')
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True) # para que en cada paso no acumule gradientes y los ponga en cero de nuevo
    loss.backward()
    optimizer.step()


context=torch.zeros((1,1),dtype=torch.long,device=device)

print(decode(model.generate(context,500)[0].tolist()))

model.parameters()