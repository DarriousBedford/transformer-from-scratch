import torch
import torch.nn as nn
from torch.nn import functional as F
from t1_tokens import encode, decode, vocab_size, train_data, val_data

torch.manual_seed(1337)   # makes the "random" results repeatable, so your numbers match each run

# ---- settings (hyperparameters) ----
BATCH_SIZE = 32       # how many chunks per training step
BLOCK_SIZE = 8        # how many characters per chunk

LEARNING_RATE = 1e-2
MAX_STEPS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 200


def get_batch(split):
    """Grab BATCH_SIZE random chunks. Return inputs (x) and targets (y = x shifted by one)."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))   # random starting positions
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])   # grab the chunks)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()                       # sets up the nn.Module machinery
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)                 # look up each char's row of scores → (B, T, C)

        if targets is None:                      # when generating text, there's no answer to compare to
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        """Start from idx (B, T) and add max_new_tokens characters, one at a time."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                                  # scores for every position
            logits = logits[:, -1, :]                              # keep only the LAST position → "what comes next?"
            probs = F.softmax(logits, dim=-1)                      # scores → probabilities
            next_idx = torch.multinomial(probs, num_samples=1)     # randomly pick one, weighted by probability
            idx = torch.cat((idx, next_idx), dim=1)                # add it to the end
        return idx

@torch.no_grad()        # no learning here, just measuring (faster, uses less memory)
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

if __name__ == "__main__":
    xb, yb = get_batch("train")
    print(xb.shape, yb.shape)
    print(xb[0])
    print(yb[0])
    print(decode(xb[0].tolist()))
    print(decode(yb[0].tolist()))
    model = BigramModel(vocab_size)
    logits, loss = model(xb, yb)
    print(logits.shape)
    print(loss.item())
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
        optimizer.zero_grad()

    print("--- AFTER training ---")
    print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))