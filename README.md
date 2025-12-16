# KeyGen - Transformer-Based Music Generator

An AI piano music generation system using PyTorch Transformers. The model learns from MIDI datasets and generates piano sequences with pitch, velocity, and duration control.

Link: https://ishaan125.github.io/TransformerPianoModel/

## 🎹 Features

- **Transformer Architecture**: Encoder-based transformer with positional encoding for sequence modeling
- **Multi-Attribute Generation**: Predicts pitch, velocity (dynamics), and duration for each note
- **Flexible Inference**: Configurable temperature, top-k sampling, and repeat penalty
- **FastAPI Server**: REST API for music generation with async model caching
- **Docker Support**: Containerized deployment with CPU and GPU options

## 📁 Project Structure

```
Piano AI/
├── api.py               # FastAPI app + Mangum handler for Lambda/API Gateway
├── load_model.py        # Model loading, sampling, post-processing
├── model.py             # TransformerNextNoteModel definition
├── output.py            # MIDI writing helpers
├── main.py              # CLI entry for preprocess/train/generate
├── preprocess.py        # Dataset preprocessing
├── train.py             # Training loop (checkpoint saving)
├── checkpoint.pth       # Trained weights (copied to /opt/model in Lambda image)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Local/container build (non-Lambda)
├── Dockerfile.lambda     # Lambda container image
├── docker-compose.yml    # Local compose (API + deps)
├── web.HTML             # Simple frontend page for load/generate calls
├── Generated Music/      # Generated MIDI outputs (local runs)
├── MIDI Datasets/        # Source MIDI data
├── Training Data/       # Prepared training splits (if used)
└── Testing Data/        # Prepared test/val splits (if used)
```

## 🚀 Quick Start

### ⚠️ Run Main to use the pretrained model checkpoint.pth to generate midi piano files to the Generated Music folder or use the link to access the website: https://ishaan125.github.io/TransformerPianoModel/

### Prerequisites

- Python 3.11+
- PyTorch 2.2.0+
- CUDA (optional, for GPU acceleration)

### Installation

```powershell
# Clone the repository
git clone https://github.com/Ishaan125/TransformerPianoModel.git
cd "Piano AI"

# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Optionally Training Your Own Model

1. **Prepare MIDI Dataset**: Place MIDI files in organized folders under `MIDI Datasets/` or use the preloaded datasets:
   ```
   Datasets/
   ├── Classical/
   │   └── Classical/  # Your .mid files here
   └── ...
   ```

2. **Preprocess Data**:
   ```powershell
   python main.py --preprocess --train-folder "Datasets/Classical/Classical"
   ```
   or set the preprocess argument to true in main

3. **Train Model**:
   ```powershell
   python main.py --train --train-folder "Datasets/Classical/Classical"
   ```
   or set the train argument to true in main
   
   Training saves checkpoints as `checkpoint.pth` (or `checkpoint_epoch_X.pth`).

4. **Generate Music**:
   ```powershell
   python main.py --generate --checkpoint checkpoint.pth --seed 60 62 64 65 67 --gen-steps 200
   ```
   or set the generate argument to true in main

### Using the API Server

#### Start Server

```powershell
# Run directly
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up
```

#### API Endpoints

**Health Check**
```bash
GET http://localhost:8000/health
```

**Load Model**
```bash
POST http://localhost:8000/load
Content-Type: application/json

{
  "checkpoint": "checkpoint.pth",
  "device": "cpu",
  "warmup": false
}
```

**Generate Music**
```bash
POST http://localhost:8000/generate
Content-Type: application/json

{
  "checkpoint": "checkpoint.pth",
  "seed": [60, 62, 64, 65, 67],
  "gen_steps": 200,
  "temperature": 1.0,
  "top_k": 5,
  "repeat_penalty": 1.3,
  "filename": "my_generated_song.mid"
}
```

Returns a MIDI file download.

**Check Status**
```bash
GET http://localhost:8000/status
```

**Unload Model**
```bash
POST http://localhost:8000/unload
Content-Type: application/json

{
  "checkpoint": "checkpoint.pth"
}
```

## 🎵 Model Architecture

### TransformerNextNoteModel

- **Input**: Sequence of pitch tokens + optional velocity/duration
- **Embedding**: Token embeddings + positional encoding
- **Encoder**: Multi-head self-attention transformer encoder (4 layers, 8 heads, 256 dim)
- **Heads**:
  - Pitch head: Classification over MIDI pitch vocabulary (0-127)
  - Velocity head: Regression for note dynamics (0-1 normalized)
  - Duration head: Regression for note length (seconds)

### Training Details

- **Loss**: Cross-entropy (pitch) + MSE (velocity/duration)
- **Optimizer**: Adam with learning rate scheduling
- **Sequence Length**: 20 tokens (configurable)
- **Batch Size**: 32 (default)
- **Epochs**: 10-50 depending on dataset size

## 🐳 Docker Deployment

### Build and Run

```powershell
# Build image
docker build -t piano-ai .

# Run container
docker run -p 8000:8000 piano-ai

# Or use docker-compose
docker-compose up -d
```

### GPU Support

```powershell
# Build with CUDA support
docker build -t piano-ai-gpu --build-arg TORCH_VERSION=2.2.0+cu118 .

# Run with GPU
docker run --gpus all -p 8000:8000 piano-ai-gpu
```

See [README_DOCKER.md](README_DOCKER.md) for detailed Docker instructions.

## 📊 Hyperparameter Tuning

### Generation Parameters

- **`seed`**: Initial sequence of MIDI pitches (e.g., `[60, 62, 64, 65, 67]` = C major scale)
- **`gen_steps`**: Number of notes to generate (50-2000)
- **`temperature`**: Sampling randomness (0.5=conservative, 1.0=balanced, 1.5=creative)
- **`top_k`**: Limit sampling to top-k most likely pitches (3-50)
- **`repeat_penalty`**: Discourage note repetition (1.0=off, 1.3=moderate, 2.0=strong)

### Training Parameters (train.py)

- **`d_model`**: Model dimension (128-512)
- **`nhead`**: Attention heads (4-16)
- **`nlayers`**: Encoder layers (2-8)
- **`learning_rate`**: Initial LR (0.0001-0.001)
- **`batch_size`**: Training batch size (16-64)

## 🔧 Resume Training

Continue from a previous checkpoint:

```python
# In train.py, load checkpoint and optimizer state
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
start_epoch = checkpoint.get('epoch', 0) + 1
```
```
## 🐛 Troubleshooting

### Common Issues

**Import Error: No module named 'torch'**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**FileNotFoundError: checkpoint.pth**
- Ensure you've trained a model first or provide correct checkpoint path
- Check `checkpoint` parameter in API requests

**CUDA out of memory**
- Reduce `batch_size` in training
- Use `device='cpu'` for inference
- Use smaller model (`d_model=128`, `nlayers=2`)

**Generated music sounds repetitive**
- Increase `repeat_penalty` (try 1.5-2.0)
- Increase `temperature` (try 1.2-1.5)
- Reduce `top_k` (try 3-5)

**Server connection refused**
- Ensure FastAPI server is running: `python -m uvicorn api:app --host 0.0.0.0 --port 8000`
- Check firewall allows port 8000

## 📝 Development

### Project Dependencies

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
torch>=2.2.0
numpy>=1.24.0
pretty_midi>=0.2.9
tqdm>=4.66.0
pydantic>=2.0.0
mangum>=0.17.0
```

### Testing

```powershell
# Test model loading
python -c "from load_model import load_model; m = load_model('checkpoint.pth')"

# Test generation
python main.py --generate --checkpoint checkpoint.pth --gen-steps 50

# Test API
python -m uvicorn api:app --reload
# Then visit http://localhost:8000/docs for Swagger UI
```
