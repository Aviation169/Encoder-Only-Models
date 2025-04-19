🤖Encoder Only Models🤖
-----
(●'◡'●) This repository contains PyTorch implementations of encoder-only transformer architectures, focused on NLP tasks. My goal is to provide lightweight yet effective transformer designs for research and experimentation.

(●'◡'●) This repository is only for encoder-only architecture models.
It will contain multiple model folders, each representing a new encoder model architecture. These are currently being tested and will be uploaded soon.

1️⃣Important Note1️⃣
-----
⚠️ Production Readiness Notice:
This is my own custom implementation of an encoder only transformer model designed for educational purposes and small-scale experimentation. For production environments required lot of data for training.

📂Available Folders📂
----
Encoder only model for sentiment analysis -FNN(Feed-Forward neural network)

Encoder only model for sentiment analysis -PNN(Progressive neural network)


🖨️Features🖨️
-----
🚀 Mini Transformer Architecture: Custom smaller transformer model with configurable layers and heads.

⚡ GPU Acceleration: Utilizes CUDA with mixed-precision training via torch.cuda.amp.

📊 Metrics Tracking: Computes accuracy and F1 score during validation and testing.

💾 Model Persistence: Includes utilities to save and load trained models.

🤗 Hugging Face Integration: Uses transformers tokenizer and datasets library for data loading.


📅Requirements📅
-----
`Python 3.7+`

`PyTorch 2.0+`

`Transformers 4.30+`

`Datasets 2.12+`

`scikit-learn`

`tqdm`

⬇️Installation⬇️
-----
`!pip install torch transformers datasets scikit-learn tqdm`

📂Usage📂
-----
You can access a free encoder-only architecture code from this repository. All the codes have been tested with training data. You can either use them directly or customize them as per your requirements.


🔮Model Configuration🔮
-----
Key architecture parameters (modifiable in code):

This is Default config;

```
config = {
    "vocab_size": tokenizer.vocab_size,  # 30522 for bert-base-uncased
    "max_len": 128,
    "hidden_size": 128,
    "num_layers": 2,
    "num_heads": 8,
    "ff_size": 512,
    "dropout": 0.1,
    "num_classes": 2
}
```

🪟Model Architecture🖼️
----
All models in our repository follow a transformer-based encoder-only architecture. Additionally, you can find encoder architectures based on FNN (Feedforward Neural Network), PNN (Progressive Neural Network), and even ViT (Vision Transformer).

**Transformer Architecture:**

1️⃣Input Embedding Layer

Converts input tokens (e.g., words, subwords) into dense vector representations.

→Often includes:

→Token embeddings

→Positional embeddings (since transformers are non-sequential)

2️⃣Multi-Head Self-Attention

→Allows the model to focus on different parts of the input simultaneously.

→Helps in capturing relationships between tokens irrespective of their distance in the sequence.

3️⃣Layer Normalization

→Applied before or after sub-layers to stabilize and speed up training.

4️⃣Feedforward Neural Network (FNN)

→A position-wise fully connected feedforward network applied to each position separately and identically.

→Usually consists of two linear transformations with a non-linear activation (e.g., ReLU or GELU).

or

4️⃣Progressive Neural Network (PNN)

→Replaces the traditional feedforward network.

→Allows knowledge accumulation across layers by progressively building representations.

→Supports lateral connections for transferring knowledge from previous tasks or layers.

5️⃣Residual Connections

→Skip connections that add the input of each sub-layer (like attention or FNN) to its output before normalization.

6️⃣Stacked Encoder Layers

→The encoder is made up of multiple identical layers (e.g., 6 or 12) containing the components above.

7️⃣Output Layer

→Outputs the final hidden states of each token.

→Can be used for various downstream tasks like classification, regression, etc.


📃License📃
----
MIT License - Free for educational and experimental use

