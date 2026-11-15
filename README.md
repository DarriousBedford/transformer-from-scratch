# Transformer from Scratch

A small GPT-style language model built step by step in PyTorch, from a simple bigram model to a 4-layer transformer, and trained on the Tiny Shakespeare dataset.

I built this to understand how large language models work on the inside, rather than only calling them through an API.

## Results

Each lesson adds one idea on top of the last. Validation loss (lower is better) dropped from **2.46 to 1.86**. A model that guesses randomly scores about 4.2.

| Lesson | File | What it adds | Val loss |
|---|---|---|---|
| T1 | `t1_tokens.py` | Character tokenizer (65 characters), encode/decode, train/val split | – |
| T2 | `t2_bigram.py` | Bigram model: predicts the next character from 1 character | 2.46 |
| T3 | `t3_attention.py` | Single-head self-attention plus position embeddings | 2.40 |
| T4 | `t4_block.py` | Transformer block: multi-head attention, feed-forward, residuals, LayerNorm (3 blocks) | 2.09 |
| T5 | `t5_gpt.py` | Full GPT: scaled up, dropout, temperature, save/load (~210k parameters) | **1.86** |

### Sample output (T5, temperature 1.0)

```
ANGHA:
Ne, be shall sofes thou camd that so thee,
Patch, be deepon, new undentatione
Yal wear with neve most plain more the tone.
```

The model learned real words, line breaks, and the script's `NAME:` + dialogue format from raw characters alone.

## How it works

```
characters → token + position embeddings
           → [ attention → feed-forward ] × 4 blocks   (residuals + LayerNorm)
           → final LayerNorm → linear layer → next-character probabilities
```

- **Self-attention** lets each character look back at earlier characters and decide which ones matter (queries, keys, values, and a causal mask so it can't see the future).
- **Multi-head attention** runs several attention heads in parallel so each one can learn a different pattern.
- **Feed-forward** layers process each position after it has gathered context.
- **Residual connections + LayerNorm** keep training stable as the network gets deeper.
- **Dropout** randomly turns off 20% of neurons during training to reduce overfitting.
- **Temperature** controls generation: below 1.0 is safer and more repetitive, above 1.0 is more creative and random.

## My GPT vs. a modern LLM

The architecture is the same. The difference is scale and training stages.

| | This project (T5) | Modern LLMs |
|---|---|---|
| Parameters | ~210,000 | hundreds of billions+ |
| Layers | 4 | ~100 |
| Context window | 32 characters | 100,000s of tokens |
| Tokens | single characters (65) | subword pieces via BPE (~100k) |
| Training data | ~1 MB of Shakespeare | trillions of tokens |
| Training stages | pretraining only | pretraining + fine-tuning + human feedback |

Because it is only pretrained, this model **continues text** instead of answering questions.

## Run it

```bash
pip install torch numpy

# download the dataset into data/
mkdir data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python t5_gpt.py
```

The first run trains on CPU (about 10–20 minutes) and saves the weights to `t5_model.pt`. Later runs load the saved model and generate right away. To retrain, delete `t5_model.pt`.

Each earlier lesson file can also be run on its own (for example `python t2_bigram.py`).

## What I learned

- How attention, embeddings, and the training loop (forward → loss → backward → step) fit together.
- How to spot overfitting by watching the train/val loss gap, and how dropout helps.
- Debugging silent bugs, for example `model.train` without `()` never switches dropout back on and raises no error.

## Credits

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Approach inspired by Andrej Karpathy's "Let's build GPT" walkthrough
- Dataset: Tiny Shakespeare, from Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
