# trainGPT-2

This repository documents the journey from understanding Transformer theory to training a fully functional GPT-2 (124M parameters) from scratch.

**The "Almost Serious" Approach:**
This isn't just a toy demo. The goal is to build a robust training pipeline by:

- **Curating Data**: Using high-quality datasets (e.g., FineWeb-Edu) instead of random text.
- **Engineering**: Implementing proper hyperparameter tuning and metric monitoring.
- **Scaling**: Leveraging GPU acceleration (MPS/CUDA) for realistic training performance.

## Phase 1: Conceptual foundation

Before coding, understand that GPT-2 is a decoder-only Transformer designed for autoregressive language modeling—predicting the next token based on all previous tokens.

- **Architecture**: 12-layer, 12-head, 768-embedding dimension (124M parameters).
- **Key Specs**: Context window of 1024 tokens and a vocabulary size of 50,257.
- **Mechanism**: Causal Self-Attention allows the model to weigh the importance of previous tokens, while masking future ones.
- **Structural Detail**: Uses Pre-Layer Normalization (Pre-LN) for training stability, unlike the original Transformer paper.
- **Objective**: Minimize the cross-entropy loss between the predicted token and the actual next token.

## Phase 2: Setting the Stage (Tools & Environment)

- **Framework**: PyTorch is recommended for its control over the training loop.
- **Hardware**: A GPU is essential.
  - **Local Development**: MacBook Pro with Apple Silicon (M1/M2/M3) using PyTorch's MPS (Metal Performance Shaders) acceleration.
  - **Cloud Notebooks**: Google Colab (Free/Pro) for prototyping on NVIDIA GPUs.
  - **Scale-up**: Dedicated GPU cloud services (e.g., Lambda Labs, RunPod, AWS) for full training runs.
- **Dataset Strategy**:
  - **Prototyping**: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (1MB) - Ideal for debugging the training loop and overfitting tests.
  - **Pre-training**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (Sample or Full) - High-quality educational content for learning actual language patterns.
  - **Fine-tuning**: Custom domain data (e.g., conversational logs in JSONL format) to specialize the model.
- **Tokenization**: Byte-level BPE, which breaks words into subword units.

## Main references:

### Educational Content

- **[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)** by Andrej Karpathy: The definitive video series for building neural networks from scratch.
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**: The original paper introducing the Transformer architecture.
- **[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**: The official GPT-2 paper from OpenAI.

### Code Repositories

- **[nanoGPT](https://github.com/karpathy/nanoGPT)**: The primary inspiration for this project. A clean, simple implementation of GPT.
- **[nanoChat](https://github.com/karpathy/nanochat)**: Useful reference for chat-specific implementations.
- **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)**: Essential for tokenizers and validating model outputs.
- **Community Implementations**: Repos like [gpt2-from-scratch](https://github.com/saqib1707/gpt2-from-scratch) provided additional context.

## Acknowledgements

- **Andrej Karpathy**: For the educational content and inspiration.
