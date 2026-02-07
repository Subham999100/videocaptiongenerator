"""
feature_extraction.py - Extract CNN Features from Video Frames

This script:
1. Loads a pre-trained InceptionV3 model (trained on ImageNet)
2. Extracts frames from each video
3. Passes frames through CNN to get feature vectors
4. Saves features for training

Why InceptionV3?
    - State-of-the-art CNN architecture
    - Pre-trained on ImageNet (1.2M images)
    - Outputs 2048-dimensional feature vectors
    - We use it as a "feature extractor" without training it
"""

import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import cv2
from utils import extract_frames
from tqdm import tqdm


class VideoFeatureExtractor:
    """
    Extract visual features from videos using pre-trained CNN.
    """
    
    def __init__(self, target_size=(299, 299)):
        """
        Initialize the feature extractor.
        
        Args:
            target_size: Input size for InceptionV3 (must be 299x299)
        
        Why 299x299?
            - InceptionV3 was trained with this input size
            - Other sizes won't work properly
        """
        self.target_size = target_size
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Load InceptionV3 model without top classification layer.
        
        What is "include_top=False"?
            - Removes final classification layer (1000 ImageNet classes)
            - Keeps only feature extraction layers
            - Output: 2048-dimensional feature vector
        
        Returns:
            InceptionV3 model for feature extraction
        """
        print("Loading InceptionV3 model...")
        
        # Load pre-trained InceptionV3 (weights from ImageNet)
        base_model = InceptionV3(
            weights='imagenet',      # Use pre-trained weights
            include_top=False,       # Remove classification layer
            pooling='avg'            # Global average pooling (2048-dim output)
        )
        
        # We don't train this model, just use it for feature extraction
        base_model.trainable = False
        
        print(f"Model loaded. Output shape: {base_model.output_shape}")
        print(f"Feature dimension: {base_model.output_shape[-1]}")
        
        return base_model
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for InceptionV3.
        
        Steps:
            1. Resize to 299x299
            2. Normalize pixel values
        
        Args:
            frame: RGB image as numpy array
        
        Returns:
            Preprocessed frame ready for CNN
        """
        # Resize to InceptionV3 input size
        frame_resized = cv2.resize(frame, self.target_size)
        
        # Convert to float and expand dimensions
        # Shape: (299, 299, 3) → (1, 299, 299, 3)
        # CNN expects batch of images, we have 1 image
        frame_array = np.expand_dims(frame_resized, axis=0)
        
        # Normalize pixel values for InceptionV3
        # Converts [0, 255] → [-1, 1] range
        frame_processed = preprocess_input(frame_array)
        
        return frame_processed
    
    def extract_features_from_frames(self, frames):
        """
        Extract features from multiple frames.
        
        Args:
            frames: List of frames (RGB images)
        
        Returns:
            Numpy array of features, shape: (num_frames, 2048)
        
        Process:
            frame1 → CNN → [0.2, 0.5, ..., 0.1] (2048 numbers)
            frame2 → CNN → [0.1, 0.3, ..., 0.4] (2048 numbers)
            ...
        """
        features = []
        
        print(f"Extracting features from {len(frames)} frames...")
        
        for frame in tqdm(frames, desc="Processing frames"):
            # Preprocess frame
            processed = self.preprocess_frame(frame)
            
            # Extract features using CNN
            # Output shape: (1, 2048)
            feature = self.model.predict(processed, verbose=0)
            
            # Remove batch dimension: (1, 2048) → (2048,)
            feature = feature.reshape(-1)
            
            features.append(feature)
        
        # Convert list to numpy array
        features_array = np.array(features)
        
        print(f"Features shape: {features_array.shape}")
        
        return features_array
    
    def extract_video_features(self, video_path, frames_per_second=1):
        """
        Complete pipeline: video → frames → features.
        
        Args:
            video_path: Path to video file
            frames_per_second: Frame extraction rate
        
        Returns:
            Single feature vector representing entire video
        
        How we represent a video:
            1. Extract N frames
            2. Get features for each frame (N x 2048)
            3. Average all frame features → single 2048-dim vector
        """
        # Step 1: Extract frames from video
        frames = extract_frames(video_path, frames_per_second=frames_per_second)
        
        if len(frames) == 0:
            print(f"Warning: No frames extracted from {video_path}")
            return None
        
        # Step 2: Extract features from each frame
        frame_features = self.extract_features_from_frames(frames)
        
        # Step 3: Average features across all frames
        # Shape: (num_frames, 2048) → (2048,)
        # This gives us one vector representing the whole video
        video_feature = np.mean(frame_features, axis=0)
        
        print(f"Final video feature shape: {video_feature.shape}")
        
        return video_feature


def extract_all_video_features(videos_dir, output_dir, frames_per_second=1):
    """
    Extract features for all videos in a directory.
    
    Args:
        videos_dir: Directory containing video files
        output_dir: Where to save feature files
        frames_per_second: Frame extraction rate
    
    Saves:
        One .npy file per video containing its feature vector
        Example: video1.mp4 → video1.npy (contains 2048 numbers)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature extractor
    extractor = VideoFeatureExtractor()
    
    # Get all video files
    video_files = [f for f in os.listdir(videos_dir) 
                   if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    print(f"\nFound {len(video_files)} videos to process")
    
    # Process each video
    for video_file in video_files:
        print(f"\n{'='*60}")
        print(f"Processing: {video_file}")
        print(f"{'='*60}")
        
        video_path = os.path.join(videos_dir, video_file)
        
        # Extract features
        features = extractor.extract_video_features(
            video_path, 
            frames_per_second=frames_per_second
        )
        
        if features is not None:
            # Save features to file
            output_filename = os.path.splitext(video_file)[0] + '.npy'
            output_path = os.path.join(output_dir, output_filename)
            
            np.save(output_path, features)
            print(f"Saved features to: {output_path}")
        else:
            print(f"Failed to extract features from {video_file}")
    
    print(f"\n{'='*60}")
    print("Feature extraction complete!")
    print(f"Features saved in: {output_dir}")
    print(f"{'='*60}")


def load_video_feature(feature_path):
    """
    Load pre-computed features from file.
    
    Args:
        feature_path: Path to .npy feature file
    
    Returns:
        Feature vector (2048-dim)
    """
    features = np.load(feature_path)
    return features


# Main execution
if __name__ == "__main__":
    """
    Run this script to extract features from all videos.
    
    Usage:
        python feature_extraction.py
    
    This will:
        1. Find all videos in data/videos/
        2. Extract frames at 1 FPS
        3. Pass frames through InceptionV3
        4. Save features to data/features/
    """
    
    # Configuration
    VIDEOS_DIR = 'data/videos'
    FEATURES_DIR = 'data/features'
    FRAMES_PER_SECOND = 1  # Extract 1 frame per second
    
    print("="*60)
    print("VIDEO FEATURE EXTRACTION")
    print("="*60)
    print(f"Videos directory: {VIDEOS_DIR}")
    print(f"Features directory: {FEATURES_DIR}")
    print(f"Extraction rate: {FRAMES_PER_SECOND} frame(s) per second")
    print("="*60)
    
    # Check if videos directory exists
    if not os.path.exists(VIDEOS_DIR):
        print(f"\nError: Directory '{VIDEOS_DIR}' not found!")
        print("Please create the directory and add video files.")
        exit(1)
    
    # Check if there are any videos
    video_files = [f for f in os.listdir(VIDEOS_DIR) 
                   if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if len(video_files) == 0:
        print(f"\nError: No video files found in '{VIDEOS_DIR}'!")
        print("Supported formats: .mp4, .avi, .mov, .mkv")
        exit(1)
    
    # Extract features
    extract_all_video_features(
        videos_dir=VIDEOS_DIR,
        output_dir=FEATURES_DIR,
        frames_per_second=FRAMES_PER_SECOND
    )
    
    print("\n" + "="*60)
    print("Next step: Run 'python train.py' to train the model")
    print("="*60)
