import torch
import torch.nn as nn
from torch.nn import functional as F
from t1_tokens import decode, vocab_size
from t2_bigram import get_batch, estimate_loss, BLOCK_SIZE
from t3_attention import Head, N_EMBD

torch.manual_seed(1337)

N_HEAD = 4      # how many attention heads run side by side

N_LAYER = 3
LEARNING_RATE = 1e-3
MAX_STEPS = 5000
EVAL_INTERVAL = 500


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, then combined."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)    # mixes the heads' results together

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """A small neural net applied to each token on its own: the 'thinking' step."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),     # expand
            nn.ReLU(),                               # activation (non-linearity)
            nn.Linear(4 * n_embd, n_embd),     # shrink back
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


class BlockModel(nn.Module):
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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
    

if __name__ == "__main__":
    model = BlockModel()
    start = torch.zeros((1, 1), dtype=torch.long)

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

    print("--- AFTER training ---")
    print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))