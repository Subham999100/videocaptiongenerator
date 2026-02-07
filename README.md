# Video Caption Generator using LSTM

A beginner-friendly deep learning project that generates captions for videos using CNN for feature extraction and LSTM for sequence generation.

## 📋 Project Overview

This project extracts features from video frames using a pre-trained CNN (InceptionV3) and generates descriptive captions using an LSTM-based decoder network.

### How It Works:
1. **Frame Extraction**: Extract frames from videos (1 frame per second)
2. **Feature Extraction**: Use InceptionV3 CNN to extract visual features
3. **Caption Processing**: Tokenize and prepare caption text
4. **Model Training**: Train LSTM decoder to generate captions
5. **Inference**: Generate captions for new videos

## 🛠️ Requirements

```bash
pip install tensorflow opencv-python numpy pandas matplotlib pillow
```

### System Requirements:
- Python 3.7+
- TensorFlow 2.x
- 4GB+ RAM (8GB recommended)
- GPU optional (recommended for faster training)

## 📁 Project Structure

```
video_caption_generator/
│
├── data/
│   ├── videos/              # Place your video files here (.mp4, .avi)
│   └── captions/            # Place your caption files here (.txt or .csv)
│
├── saved_models/            # Trained models saved here
│
├── utils.py                 # Helper functions (frame extraction, text processing)
├── feature_extraction.py    # Extract features from video frames using CNN
├── model.py                 # LSTM model architecture
├── train.py                 # Training script
├── inference.py             # Generate captions for new videos
└── README.md                # This file
```

## 📊 Dataset Format

### Option 1: Text Files
```
data/
├── videos/
│   ├── video1.mp4
│   └── video2.mp4
└── captions/
    ├── video1.txt  (contains: "A person playing guitar in a park")
    └── video2.txt  (contains: "A cat jumping over a fence")
```

### Option 2: CSV File
Create `data/captions/captions.csv`:
```csv
video_name,caption
video1.mp4,A person playing guitar in a park
video2.mp4,A cat jumping over a fence
```

## 🚀 Usage

### Step 1: Prepare Your Data
1. Place video files in `data/videos/`
2. Create corresponding caption files in `data/captions/`

### Step 2: Extract Features
```bash
python feature_extraction.py
```
This extracts frames and generates CNN features for all videos.

### Step 3: Train the Model
```bash
python train.py
```
Trains the LSTM model to generate captions (this may take time).

### Step 4: Generate Captions
```bash
python inference.py --video_path data/videos/test_video.mp4
```

## 🧠 Model Architecture

```
Input Video
    ↓
[Frame Extraction] → Extract 1 frame/second
    ↓
[InceptionV3 CNN] → Extract 2048-dim features per frame
    ↓
[Average Pooling] → Single 2048-dim vector per video
    ↓
[Dense Layer] → Project to 256 dimensions
    ↓
[LSTM Decoder] → Generate caption word by word
    ↓
Generated Caption
```

### Model Components:
- **CNN Encoder**: InceptionV3 (pre-trained on ImageNet)
- **Feature Dimension**: 2048 → 256
- **LSTM Units**: 256
- **Embedding Dimension**: 256
- **Vocabulary**: Generated from training captions

## 📝 Key Concepts

### 1. Frame Extraction
- Extract frames at 1 FPS (1 frame per second)
- Resize frames to 299x299 (InceptionV3 input size)
- Convert to RGB format

### 2. Feature Extraction
- Use InceptionV3 without top classification layer
- Extract 2048-dimensional feature vector per frame
- Average features across all frames for video representation

### 3. Caption Processing
- Add `<start>` and `<end>` tokens to captions
- Convert words to integers (tokenization)
- Pad sequences to same length

### 4. LSTM Training
- Teacher forcing: use ground truth previous word
- Categorical cross-entropy loss
- Adam optimizer

### 5. Caption Generation
- Start with `<start>` token
- Generate one word at a time
- Stop at `<end>` token or max length

## 🎯 Training Tips

1. **Small Dataset**: Start with 10-20 videos for testing
2. **Epochs**: 20-50 epochs (watch for overfitting)
3. **Batch Size**: 32-64 (adjust based on RAM)
4. **Learning Rate**: 0.001 (default Adam)

## 🔍 Expected Results

With a small dataset (10-20 videos):
- Training may overfit
- Captions will be generic but grammatically correct
- Example: "a person is doing something"

For better results:
- Use 1000+ diverse videos
- Train for longer
- Tune hyperparameters

## 📚 Understanding the Code

### utils.py
- `extract_frames()`: Gets frames from video
- `load_captions()`: Loads caption data
- `create_tokenizer()`: Builds vocabulary from captions

### feature_extraction.py
- Loads InceptionV3 model
- Processes video frames
- Saves features as .npy files

### model.py
- Defines LSTM architecture
- Image features → Dense → LSTM → Dense → Softmax

### train.py
- Creates data generators
- Trains the model
- Saves model and tokenizer

### inference.py
- Loads trained model
- Generates captions for new videos
- Uses greedy decoding

## 🐛 Troubleshooting

**Error: Out of Memory**
- Reduce batch size
- Use fewer frames per video
- Process videos one at a time

**Error: No module named 'cv2'**
```bash
pip install opencv-python
```

**Poor Caption Quality**
- Train with more data
- Increase training epochs
- Check caption quality in dataset

## 🎓 For Viva/Presentation

### Key Points to Explain:
1. **Why CNN + LSTM?**
   - CNN: Extract visual features (what's in the frame)
   - LSTM: Generate sequential text (word by word)

2. **What is Transfer Learning?**
   - Using pre-trained InceptionV3 (trained on ImageNet)
   - Don't train CNN, only train LSTM decoder

3. **How does LSTM generate captions?**
   - Maintains hidden state (memory)
   - Generates one word at a time
   - Uses previous word to predict next word

4. **Training Process:**
   - Feed image features + partial caption
   - Predict next word
   - Compare with actual next word
   - Adjust weights to minimize error

## 📖 References

- InceptionV3: Rethinking the Inception Architecture (Szegedy et al., 2015)
- Show and Tell: A Neural Image Caption Generator (Vinyals et al., 2015)
- LSTM: Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created for college deep learning project

---

**Note**: This is a simplified implementation for learning purposes. Production systems use more sophisticated architectures like attention mechanisms and beam search decoding.
