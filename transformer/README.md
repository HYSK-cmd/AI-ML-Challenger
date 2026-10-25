# Train a small GPT from scratch

A Jupyter notebook that trains a small GPT-style, decoder-only language model with PyTorch.
Based on [kevinpdev/gpt-from-scratch](https://github.com/kevinpdev/gpt-from-scratch) (MIT).

- `llm-from-scratch.ipynb` – tokenization, positional encoding, self-attention, decoder blocks, pretraining, SFT, inference
- `GPTMaster.ipynb` – another training notebook (WikiText pretraining)
- `pretrain_final.pth`, `sft_final.pth` – trained weights (Git LFS)
- Setup: `pip install -r requirements.txt`, then open the notebook.
