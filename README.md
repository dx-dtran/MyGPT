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

### GPT?

* [transformer.py](MyGPT/transformer.py) is an implementation of the Transformer decoder architecture described in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper
* [pretrain.py](MyGPT/pretrain.py) performs a training loop that improves MyGPT's ability to predict tokens. Output text is periodically sampled during training to help visualize predictive ability over time
* [generate.py](MyGPT/generate.py) takes a prompt, feeds it into a pre-trained MyGPT model, and generates text
