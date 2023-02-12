# MyGPT

My implementation of a small language model in PyTorch

Given a sequence of tokens, MyGPT predicts the next token

```python
vocab = ["cat", "hat", "the", "in"]
mygpt = MyGPT(vocab)
prediction = mygpt(["the", "cat", "in", "the"])

# prediction = "hat"
```
