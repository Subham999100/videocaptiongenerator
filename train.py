"""
train.py - Train the Video Caption Generator

This script:
1. Loads video features and captions
2. Prepares training data
3. Trains the LSTM model
4. Saves the trained model

Training Process:
    For each video:
        - Get image features
        - Get caption: "<start> a dog running <end>"
        - Create training pairs:
            Input: features + "<start>"          → Output: "a"
            Input: features + "<start> a"        → Output: "dog"
            Input: features + "<start> a dog"    → Output: "running"
            ...
    
    This is called "teacher forcing" - we give correct previous words during training.
"""

import os
import numpy as np
from model import create_caption_model, compile_model, model_summary
from utils import (load_captions_from_txt, preprocess_captions, 
                   create_tokenizer, get_max_caption_length,
                   save_tokenizer)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import pickle


class DataGenerator:
    """
    Generate training batches for the model.
    
    Why do we need this?
        - Can't load all data into memory at once
        - Generate data on-the-fly during training
        - Each epoch can shuffle data differently
    """
    
    def __init__(self, features_dict, captions_dict, tokenizer, max_length, 
                 vocab_size, batch_size=32):
        """
        Initialize the data generator.
        
        Args:
            features_dict: {video_name: features_array}
            captions_dict: {video_name: caption_text}
            tokenizer: Fitted tokenizer
            max_length: Maximum caption length
            vocab_size: Vocabulary size
            batch_size: Number of samples per batch
        """
        self.features_dict = features_dict
        self.captions_dict = captions_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        
        # Get list of video names
        self.video_names = list(captions_dict.keys())
        self.num_samples = len(self.video_names)
    
    def generate_batch(self):
        """
        Generate training batches infinitely.
        
        Yields:
            ([image_features, text_sequences], target_words)
        
        Example of one training sample:
            Image features: [0.2, 0.5, ..., 0.1] (2048 numbers)
            Input sequence: [2, 5, 12, 0, 0] ("<start> a dog" + padding)
            Target: 34 (word index for "running")
        """
        while True:
            # Shuffle videos each epoch
            np.random.shuffle(self.video_names)
            
            for i in range(0, self.num_samples, self.batch_size):
                # Get batch of video names
                batch_videos = self.video_names[i:i + self.batch_size]
                
                # Lists to store batch data
                batch_features = []
                batch_sequences = []
                batch_targets = []
                
                for video_name in batch_videos:
                    # Get features for this video
                    features = self.features_dict[video_name]
                    
                    # Get caption for this video
                    caption = self.captions_dict[video_name]
                    
                    # Convert caption to sequence of integers
                    caption_seq = self.tokenizer.texts_to_sequences([caption])[0]
                    
                    # Create training pairs from this caption
                    # Caption: [2, 5, 12, 34, 3] (<start> a dog running <end>)
                    # Create:
                    #   Input: [2, 0, 0, 0, 0]           → Target: 5
                    #   Input: [2, 5, 0, 0, 0]           → Target: 12
                    #   Input: [2, 5, 12, 0, 0]          → Target: 34
                    #   Input: [2, 5, 12, 34, 0]         → Target: 3
                    
                    for j in range(1, len(caption_seq)):
                        # Input sequence: from start to current position
                        input_seq = caption_seq[:j]
                        
                        # Pad to max_length
                        input_seq_padded = np.zeros(self.max_length)
                        input_seq_padded[:len(input_seq)] = input_seq
                        
                        # Target: next word
                        target_word = caption_seq[j]
                        
                        # Add to batch
                        batch_features.append(features)
                        batch_sequences.append(input_seq_padded)
                        batch_targets.append(target_word)
                
                # Convert to numpy arrays
                X_features = np.array(batch_features)
                X_sequences = np.array(batch_sequences)
                y_targets = np.array(batch_targets)
                
                yield ((X_features, X_sequences), y_targets)
    
    def get_steps_per_epoch(self):
        """
        Calculate number of batches per epoch.
        
        Returns:
            Number of steps
        """
        total_sequences = 0
        for caption in self.captions_dict.values():
            caption_seq = self.tokenizer.texts_to_sequences([caption])[0]
            total_sequences += len(caption_seq) - 1  # -1 because we predict next word
        
        steps = total_sequences // self.batch_size
        return max(steps, 1)  # At least 1 step


def load_all_features(features_dir, video_names):
    """
    Load pre-computed features for all videos.
    
    Args:
        features_dir: Directory containing .npy feature files
        video_names: List of video filenames
    
    Returns:
        Dictionary: {video_name: features_array}
    """
    features_dict = {}
    
    for video_name in video_names:
        # Get feature filename
        feature_filename = os.path.splitext(video_name)[0] + '.npy'
        feature_path = os.path.join(features_dir, feature_filename)
        
        if os.path.exists(feature_path):
            features = np.load(feature_path)
            features_dict[video_name] = features
        else:
            print(f"Warning: Features not found for {video_name}")
    
    print(f"Loaded features for {len(features_dict)} videos")
    return features_dict


def train_model(videos_dir, features_dir, captions_dir, model_save_dir,
                epochs=20, batch_size=32, learning_rate=0.001):
    """
    Complete training pipeline.
    
    Args:
        videos_dir: Directory with video files
        features_dir: Directory with extracted features
        captions_dir: Directory with caption files
        model_save_dir: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    
    print("="*60)
    print("VIDEO CAPTION GENERATOR - TRAINING")
    print("="*60)
    
    # ==================== STEP 1: LOAD CAPTIONS ====================
    print("\nStep 1: Loading captions...")
    
    # Try to load from txt files
    captions_dict = load_captions_from_txt(captions_dir, videos_dir)
    
    if len(captions_dict) == 0:
        print("Error: No captions found!")
        return
    
    # Preprocess captions (add start/end tokens)
    captions_dict = preprocess_captions(captions_dict)
    
    print(f"Loaded {len(captions_dict)} captions")
    print("\nExample caption:")
    example_video = list(captions_dict.keys())[0]
    print(f"  {example_video}: {captions_dict[example_video]}")
    
    
    # ==================== STEP 2: CREATE TOKENIZER ====================
    print("\nStep 2: Creating tokenizer...")
    
    # Get all captions
    all_captions = list(captions_dict.values())
    
    # Create tokenizer
    tokenizer = create_tokenizer(all_captions, max_words=5000)
    vocab_size = len(tokenizer.word_index) + 1
    
    # Get maximum caption length
    max_length = get_max_caption_length(all_captions, tokenizer)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Maximum caption length: {max_length}")
    
    
    # ==================== STEP 3: LOAD FEATURES ====================
    print("\nStep 3: Loading video features...")
    
    video_names = list(captions_dict.keys())
    features_dict = load_all_features(features_dir, video_names)
    
    if len(features_dict) == 0:
        print("Error: No features found!")
        print("Please run 'python feature_extraction.py' first.")
        return
    
    
    # ==================== STEP 4: CREATE MODEL ====================
    print("\nStep 4: Creating model...")
    
    model = create_caption_model(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=256,
        lstm_units=256,
        feature_dim=2048
    )
    
    model = compile_model(model, learning_rate=learning_rate)
    model_summary(model)
    
    
    # ==================== STEP 5: CREATE DATA GENERATOR ====================
    print("\nStep 5: Setting up data generator...")
    
    data_gen = DataGenerator(
        features_dict=features_dict,
        captions_dict=captions_dict,
        tokenizer=tokenizer,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=batch_size
    )
    
    steps_per_epoch = data_gen.get_steps_per_epoch()
    print(f"Steps per epoch: {steps_per_epoch}")
    
    
    # ==================== STEP 6: SETUP CALLBACKS ====================
    print("\nStep 6: Setting up training callbacks...")
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Callback 1: Save best model
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_dir, 'best_model.keras'),
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    
    # Callback 2: Early stopping (stop if no improvement)
    early_stop = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Callback 3: Reduce learning rate if stuck
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    
    # ==================== STEP 7: TRAIN MODEL ====================
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = model.fit(
        data_gen.generate_batch(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    
    # ==================== STEP 8: SAVE MODEL AND TOKENIZER ====================
    print("\n" + "="*60)
    print("SAVING MODEL AND TOKENIZER")
    print("="*60)
    
    # Save final model
    model.save(os.path.join(model_save_dir, 'final_model.keras'))
    print(f"Model saved to {model_save_dir}/final_model.keras")
    
    # Save tokenizer
    tokenizer_path = os.path.join(model_save_dir, 'tokenizer.pkl')
    save_tokenizer(tokenizer, tokenizer_path)
    
    # Save training configuration
    config = {
        'vocab_size': vocab_size,
        'max_length': max_length,
        'embedding_dim': 256,
        'lstm_units': 256,
        'feature_dim': 2048
    }
    
    config_path = os.path.join(model_save_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"Config saved to {config_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTrained for {len(history.history['loss'])} epochs")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"\nNext step: Run 'python inference.py' to generate captions")
    print("="*60)


# Main execution
if __name__ == "__main__":
    """
    Run training.
    
    Usage:
        python train.py
    """
    
    # Configuration
    VIDEOS_DIR = 'data/videos'
    FEATURES_DIR = 'data/features'
    CAPTIONS_DIR = 'data/captions'
    MODEL_SAVE_DIR = 'saved_models'
    
    # Training hyperparameters
    EPOCHS = 20          # Number of training iterations
    BATCH_SIZE = 32      # Samples per batch
    LEARNING_RATE = 0.001  # Step size for weight updates
    
    print("\nTraining Configuration:")
    print(f"  Videos: {VIDEOS_DIR}")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Captions: {CAPTIONS_DIR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print()
    
    # Check directories exist
    if not os.path.exists(FEATURES_DIR):
        print(f"Error: Features directory '{FEATURES_DIR}' not found!")
        print("Please run 'python feature_extraction.py' first.")
        exit(1)
    
    if not os.path.exists(CAPTIONS_DIR):
        print(f"Error: Captions directory '{CAPTIONS_DIR}' not found!")
        print("Please create caption files in the captions directory.")
        exit(1)
    
    # Start training
    train_model(
        videos_dir=VIDEOS_DIR,
        features_dir=FEATURES_DIR,
        captions_dir=CAPTIONS_DIR,
        model_save_dir=MODEL_SAVE_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
