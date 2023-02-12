# MyGPT

My implementation of a small language model in PyTorch

Given a sequence of tokens, MyGPT predicts the next token

```python
vocab = ["cat", "hat", "the", "in"]

mygpt = MyGPT(vocab)
prediction = mygpt(["the", "cat", "in", "the"])

# prediction = "hat"
```

See the [output folder](https://github.com/dx-dtran/MyGPT/tree/main/output) for sample text MyGPT produced
