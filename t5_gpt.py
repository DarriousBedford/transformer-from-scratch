import torch
import torch.nn as nn
import os
from torch.nn import functional as F
from t1_tokens import decode, vocab_size, train_data, val_data

torch.manual_seed(1337)

# --- settings (scaled up from T4) ---
BATCH_SIZE = 32
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
MODEL_PATH = "t5_model.pt"


def get_batch(split):
    """Grab BATCH_SIZE random chunks. Return inputs (x) and targets (y = x shifted by one)."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))   # random starting positions
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])   # grab the chunks)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model):
    """Average the loss over EVAL_ITERS batches, for both train and val."""
        
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()

    
    return out
    
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)   # "what do I contain?"
        self.query = nn.Linear(N_EMBD, head_size, bias=False)   # "what am I looking for?"
        self.value = nn.Linear(N_EMBD, head_size, bias=False)   # "what I'll share"
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))    
        self.dropout = nn.Dropout(DROPOUT)          # NEW

    def forward(self, x):
        B, T, C = x.size() # batch size, time steps, embedding dimensionality (n_embd)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # prevent attending to future tokens
        wei = F.softmax(wei, dim=-1)                    # (B, T, T)
        wei = self.dropout(wei)                         # NEW
        out = wei @ v                                    # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, then combined."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)    # mixes the heads' results together
        self.dropout = nn.Dropout(DROPOUT)          # NEW

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))
    



class FeedForward(nn.Module):
    """A small neural net applied to each token on its own: the 'thinking' step."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),     # expand
            nn.ReLU(),                               # activation (non-linearity)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)  # shrink back
        )


    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """One transformer block: communicate (attention), then think (feed-forward)."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)      # normalize before attention
        self.ln2 = nn.LayerNorm(n_embd)      # normalize before feed-forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # communicate (+ residual)
        x = x + self.ffwd(self.ln2(x))    # think (+ residual)
        return x

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])   # NEW
        self.ln_f = nn.LayerNorm(N_EMBD)                                   # NEW
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T))
        x = tok + pos
        x = self.blocks(x)        # NEW
        x = self.ln_f(x)          # NEW
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, idx, max_new_tokens, temperature=1.0):     # NEW parameter
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # keep only the last position, then divide by temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


if __name__ == "__main__":
    model = GPTModel()
    print(sum(p.numel() for p in model.parameters()), "parameters")

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        for step in range(MAX_STEPS):
            if step % EVAL_INTERVAL == 0:
                losses = estimate_loss(model)
                print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

            xb, yb = get_batch("train")
            optimizer.zero_grad()
            logits, loss = model(xb, yb)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved to", MODEL_PATH)

    model.eval()    # dropout OFF for generating
    start = torch.zeros((1, 1), dtype=torch.long)
    for temp in [0.5, 1.0, 1.5]:
        print(f"\n--- temperature {temp} ---")
        print(decode(model.generate(start, max_new_tokens=300, temperature=temp)[0].tolist()))