"""
utils.py - Utility Functions for Video Caption Generator

This module contains helper functions for:
1. Extracting frames from videos
2. Loading and processing captions
3. Creating tokenizers for text
4. Padding sequences
"""

import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


def extract_frames(video_path, frames_per_second=1, max_frames=None):
    """
    Extract frames from a video file at specified rate.
    
    Args:
        video_path: Path to video file
        frames_per_second: How many frames to extract per second (default: 1)
        max_frames: Maximum number of frames to extract (default: None)
    
    Returns:
        List of frames as numpy arrays
    
    Example:
        frames = extract_frames('video.mp4', frames_per_second=1)
        # Returns ~30 frames for a 30-second video
    """
    # Open video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)  # Original FPS of video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")
    
    # Calculate frame interval
    # If video is 30 FPS and we want 1 frame/second, we skip every 30 frames
    frame_interval = int(fps / frames_per_second)
    
    frames = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        # Read next frame
        success, frame = video.read()
        
        if not success:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Convert BGR (OpenCV format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted_count += 1
            
            # Stop if we've reached max_frames
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    # Release video
    video.release()
    
    print(f"Extracted {len(frames)} frames from {video_path}")
    
    return frames


def load_captions_from_txt(captions_dir, videos_dir):
    """
    Load captions from individual text files.
    
    Assumes:
        - Each video has a corresponding .txt file with same name
        - data/videos/video1.mp4 → data/captions/video1.txt
    
    Args:
        captions_dir: Directory containing caption text files
        videos_dir: Directory containing video files
    
    Returns:
        Dictionary: {video_filename: caption_text}
    
    Example:
        captions = load_captions_from_txt('data/captions', 'data/videos')
        # Returns: {'video1.mp4': 'A person walking in the park', ...}
    """
    captions_dict = {}
    
    # Get all video files
    video_files = [f for f in os.listdir(videos_dir) 
                   if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        # Get corresponding caption file
        video_name = os.path.splitext(video_file)[0]
        caption_file = os.path.join(captions_dir, video_name + '.txt')
        
        if os.path.exists(caption_file):
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
                captions_dict[video_file] = caption
        else:
            print(f"Warning: No caption file found for {video_file}")
    
    print(f"Loaded {len(captions_dict)} captions")
    return captions_dict


def load_captions_from_csv(csv_path):
    """
    Load captions from a CSV file.
    
    CSV format:
        video_name,caption
        video1.mp4,"A person playing guitar"
        video2.mp4,"A cat jumping"
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Dictionary: {video_filename: caption_text}
    """
    df = pd.read_csv(csv_path)
    captions_dict = dict(zip(df['video_name'], df['caption']))
    
    print(f"Loaded {len(captions_dict)} captions from CSV")
    return captions_dict


def preprocess_captions(captions_dict):
    """
    Preprocess captions by adding start and end tokens.
    
    Why we add tokens:
        - <start>: Tells model when caption begins
        - <end>: Tells model when caption ends
    
    Args:
        captions_dict: Dictionary of {video_name: caption}
    
    Returns:
        Dictionary with processed captions
    
    Example:
        Input: {'video1.mp4': 'A dog running'}
        Output: {'video1.mp4': '<start> a dog running <end>'}
    """
    processed_captions = {}
    
    for video_name, caption in captions_dict.items():
        # Convert to lowercase for consistency
        caption = caption.lower()
        
        # Add start and end tokens
        caption = '<start> ' + caption + ' <end>'
        
        processed_captions[video_name] = caption
    
    return processed_captions


def create_tokenizer(captions_list, max_words=5000):
    """
    Create a tokenizer to convert words to integers.
    
    What is tokenization?
        - Converts words to numbers (e.g., 'dog' → 5, 'cat' → 12)
        - Model works with numbers, not words
    
    Args:
        captions_list: List of all captions
        max_words: Maximum vocabulary size
    
    Returns:
        Fitted Tokenizer object
    
    Example:
        captions = ['a dog', 'a cat']
        tokenizer = create_tokenizer(captions)
        # tokenizer can now convert words ↔ numbers
    """
    tokenizer = Tokenizer(num_words=max_words, 
                         oov_token='<unk>',  # For unknown words
                         filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    
    # Learn vocabulary from all captions
    tokenizer.fit_on_texts(captions_list)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    return tokenizer


def caption_to_sequence(caption, tokenizer):
    """
    Convert a caption to a sequence of integers.
    
    Args:
        caption: Text caption
        tokenizer: Fitted tokenizer
    
    Returns:
        List of integers
    
    Example:
        caption = '<start> a dog running <end>'
        seq = caption_to_sequence(caption, tokenizer)
        # Returns: [2, 5, 12, 34, 3] (numbers for each word)
    """
    sequence = tokenizer.texts_to_sequences([caption])[0]
    return sequence


def get_max_caption_length(captions_list, tokenizer):
    """
    Find the maximum caption length in the dataset.
    
    Why we need this:
        - All sequences must be same length for batch processing
        - We'll pad shorter sequences to this length
    
    Args:
        captions_list: List of all captions
        tokenizer: Fitted tokenizer
    
    Returns:
        Maximum sequence length
    """
    max_length = 0
    
    for caption in captions_list:
        sequence = tokenizer.texts_to_sequences([caption])[0]
        max_length = max(max_length, len(sequence))
    
    print(f"Maximum caption length: {max_length} words")
    return max_length


def pad_sequence_to_length(sequence, max_length):
    """
    Pad sequence to specified length with zeros.
    
    Why padding?
        - Neural networks need fixed-size inputs
        - Short sequences get zeros added to end
    
    Args:
        sequence: List of integers
        max_length: Target length
    
    Returns:
        Padded sequence
    
    Example:
        seq = [2, 5, 12]
        padded = pad_sequence_to_length(seq, max_length=5)
        # Returns: [2, 5, 12, 0, 0]
    """
    padded = pad_sequences([sequence], 
                          maxlen=max_length, 
                          padding='post',  # Add zeros at end
                          truncating='post')[0]
    return padded


def save_tokenizer(tokenizer, filepath):
    """
    Save tokenizer to disk for later use.
    
    Args:
        tokenizer: Tokenizer object
        filepath: Where to save
    """
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {filepath}")


def load_tokenizer(filepath):
    """
    Load tokenizer from disk.
    
    Args:
        filepath: Path to saved tokenizer
    
    Returns:
        Tokenizer object
    """
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {filepath}")
    return tokenizer


def sequence_to_caption(sequence, tokenizer):
    """
    Convert sequence of integers back to text caption.
    
    Args:
        sequence: List of integers
        tokenizer: Fitted tokenizer
    
    Returns:
        Text caption
    
    Example:
        seq = [2, 5, 12, 34, 3]
        caption = sequence_to_caption(seq, tokenizer)
        # Returns: '<start> a dog running <end>'
    """
    # Create reverse word index (number → word)
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    # Convert each number to word
    words = []
    for num in sequence:
        if num != 0:  # Skip padding
            word = reverse_word_index.get(num, '<unk>')
            words.append(word)
    
    caption = ' '.join(words)
    return caption


def clean_caption(caption):
    """
    Remove start and end tokens from generated caption.
    
    Args:
        caption: Caption with tokens
    
    Returns:
        Clean caption
    
    Example:
        Input: '<start> a dog running <end>'
        Output: 'a dog running'
    """
    caption = caption.replace('<start>', '').replace('<end>', '').strip()
    return caption


# Example usage
if __name__ == "__main__":
    # Test frame extraction
    print("Testing frame extraction...")
    # frames = extract_frames('sample_video.mp4', frames_per_second=1)
    
    # Test caption loading
    print("\nTesting caption loading...")
    # captions = load_captions_from_txt('data/captions', 'data/videos')
    # processed = preprocess_captions(captions)
    
    # Test tokenization
    print("\nTesting tokenization...")
    sample_captions = [
        '<start> a person walking in the park <end>',
        '<start> a dog running on the beach <end>',
        '<start> a cat sitting on a chair <end>'
    ]
    
    tokenizer = create_tokenizer(sample_captions)
    
    # Convert caption to sequence
    seq = caption_to_sequence(sample_captions[0], tokenizer)
    print(f"Sequence: {seq}")
    
    # Convert back to caption
    reconstructed = sequence_to_caption(seq, tokenizer)
    print(f"Reconstructed: {reconstructed}")
    
    print("\nUtils module working correctly!")
