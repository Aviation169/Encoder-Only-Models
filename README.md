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
Encoder only model for sentiment analysis -FNN


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


📃License📃
----
MIT License - Free for educational and experimental use

