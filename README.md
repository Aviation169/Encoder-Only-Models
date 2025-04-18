🤖Encoder Only Models🤖
-----
(●'◡'●) This repository contains a PyTorch implementation of a compact of enoder only transformer-based model for sentiment analysis on the IMDB movie review dataset. The model is designed to be lightweight while maintaining competitive performance.

1️⃣Important Note1️⃣
-----
⚠️ Production Readiness Notice:
This is my own custom implementation of an encoder only transformer model designed for educational purposes and small-scale experimentation. For production environments required lot of data for training.

🖨️Features🖨️
-----
🚀 Mini Transformer Architecture: Custom smaller transformer model with configurable layers and heads.

⚡ GPU Acceleration: Utilizes CUDA with mixed-precision training via torch.cuda.amp.

📊 Metrics Tracking: Computes accuracy and F1 score during validation and testing.

💾 Model Persistence: Includes utilities to save and load trained models.

🤗 Hugging Face Integration: Uses transformers tokenizer and datasets library for data loading.

📅Requirements📅
-----
Python 3.7+

PyTorch 2.0+

Transformers 4.30+

Datasets 2.12+

scikit-learn

tqdm

⬇️Installation⬇️
-----
`!pip install torch transformers datasets scikit-learn tqdm`

📂Usage📂
-----
Training the Model
Run the training script:
`python train.py`

^_^First-run dataset setup:

Automatically downloads and caches IMDB dataset (~80MB)

Subsequent runs use local cached version

^_^Expected output:

GPU detection and VRAM information

Training progress bar with loss metrics

Validation metrics after each epoch

Final test set evaluation

🔮Model Configuration🔮
-----
Key architecture parameters (modifiable in code):

`config = {
    "vocab_size": tokenizer.vocab_size,  # 30522 for bert-base-uncased
    "max_len": 128,
    "hidden_size": 128,
    "num_layers": 2,
    "num_heads": 8,
    "ff_size": 512,
    "dropout": 0.1,
    "num_classes": 2
}`

🪟Model Architecture🖼️
----
MiniTransformerEncoder components:


|Component|Specification|
---------------|--------------
|Hidden Size	|128|
|Transformer Layers|	2|
|Attention Heads |	8|
|FFN Dimension	|512|
|Total Parameters	|~1.4M|

😊Expected Performance😊
-----

|Metric	|Value|
--------|-----
|Accuracy|	~64%|
|F1 Score|	~67%|
|Training Time|	<5min (on RTX 4080 GPU)|



🔽Saved Models🔽
----
→Trained models are saved as:

`sentiment_model.pth`

→To load a trained model:

`from train import load_model, MiniTransformerEncoder
config = {...}  # Same as training configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("sentiment_model.pth", config, device)`

📃License📃
----
MIT License - Free for educational and experimental use

