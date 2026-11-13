import torch
import torch.nn as nn
from torch.nn import functional as F
from t1_tokens import decode, vocab_size
from t2_bigram import get_batch, estimate_loss, BLOCK_SIZE

torch.manual_seed(1337)

N_EMBD = 32        # size of each token's vector (its "meaning" numbers)


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)   # "what do I contain?"
        self.query = nn.Linear(N_EMBD, head_size, bias=False)   # "what am I looking for?"
        self.value = nn.Linear(N_EMBD, head_size, bias=False)   # "what I'll share"
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))   # the triangle mask

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # 1. SCORES: how well each Query matches each Key  (this replaces the zeros from the demo!)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # 2. MASK the future + SOFTMAX → weights
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        # 3. Weighted average of the VALUES
        v = self.value(x)
        out = wei @ v
        return out
LEARNING_RATE = 1e-3     # smaller nudges than T2: attention is more sensitive
MAX_STEPS = 5000
EVAL_INTERVAL = 500


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)       # each char → 32 numbers
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)    # each position (0–7) → 32 numbers
        self.sa_head = Head(N_EMBD)                                   # one self-attention head
        self.lm_head = nn.Linear(N_EMBD, vocab_size)                  # 32 numbers → 65 scores

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                   # (B, T, 32)  WHAT each char is
        pos = self.position_embedding(torch.arange(T))    # (T, 32)     WHERE each char is
        x = tok + pos                  # (B, T, 32)
        x = self.sa_head(x)          # (B, T, 32)  gather context
        logits = self.lm_head(x)             # (B, T, 65)

                # look up each char's row of scores → (B, T, C)
       
        if targets is None:                      # when generating text, there's no answer to compare to
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]                # only the last 8 chars (positions only go 0–7!)
            logits, _ = self(idx_cond)                                # scores for every position
            logits = logits[:, -1, :]                              # keep only the LAST position → "what comes next?"
            probs = F.softmax(logits, dim=-1)                      # scores → probabilities
            next_idx = torch.multinomial(probs, num_samples=1)     # randomly pick one, weighted by probability
            idx = torch.cat((idx, next_idx), dim=1)  
        return idx


if __name__ == "__main__":
    model = AttentionModel()
    start = torch.zeros((1, 1), dtype=torch.long)     # start from a single newline character (id 0)
    print("--- BEFORE training ---")
    print(decode(model.generate(start, max_new_tokens=200)[0].tolist()))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for step in range(MAX_STEPS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
    
        xb, yb = get_batch("train"  )
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("--- AFTER training ---")
    print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))