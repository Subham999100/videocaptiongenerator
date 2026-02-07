"""
inference.py - Generate Captions for New Videos

This script:
1. Loads trained model and tokenizer
2. Extracts features from a new video
3. Generates caption word by word
4. Returns the final caption
"""

import os
import numpy as np
import argparse
import pickle

from tensorflow.keras.models import load_model
from utils import load_tokenizer, sequence_to_caption, clean_caption
from feature_extraction import VideoFeatureExtractor


class CaptionGenerator:
    """
    Generate captions for videos using trained model.
    """

    def __init__(self, model_path, tokenizer_path, config_path):
        print("Loading model and tokenizer...")

        # Load model
        self.model = load_model(model_path)
        print(f"✓ Model loaded from {model_path}")

        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)
        print(f"✓ Tokenizer loaded from {tokenizer_path}")

        # Load configuration
        with open(config_path, "rb") as f:
            self.config = pickle.load(f)

        self.vocab_size = self.config["vocab_size"]
        self.max_length = self.config["max_length"]

        print("✓ Configuration loaded")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Max caption length: {self.max_length}")

        # Initialize feature extractor
        self.feature_extractor = VideoFeatureExtractor()
        print("✓ Feature extractor initialized")

    def generate_caption_greedy(self, video_features):
        """
        Generate caption using greedy decoding.
        """

        # Start with <start> token
        start_index = self.tokenizer.word_index.get("<start>", 1)
        caption_sequence = [start_index]

        for _ in range(self.max_length):

            # Prepare text input
            input_seq = np.zeros(self.max_length)
            input_seq[:len(caption_sequence)] = caption_sequence
            input_seq = np.expand_dims(input_seq, axis=0)

            # Prepare image features
            features = np.expand_dims(video_features, axis=0)

            # Predict next word (shape: (1, vocab_size))
            predictions = self.model.predict([features, input_seq], verbose=0)

            predicted_word_index = np.argmax(predictions[0])

            # Stop if <end> token
            end_index = self.tokenizer.word_index.get("<end>", 0)
            if predicted_word_index == end_index:
                break

            caption_sequence.append(predicted_word_index)

        # Convert indices to words
        caption = sequence_to_caption(caption_sequence, self.tokenizer)
        caption = clean_caption(caption)

        return caption

    def generate_caption_for_video(self, video_path, frames_per_second=1):
        """
        Complete pipeline: video → features → caption.
        """
        print(f"\nProcessing video: {video_path}")
        print("Extracting features...")

        video_features = self.feature_extractor.extract_video_features(
            video_path, frames_per_second=frames_per_second
        )

        if video_features is None:
            print("Error: Could not extract features")
            return None

        print("Generating caption...")
        caption = self.generate_caption_greedy(video_features)
        return caption


def generate_caption(video_path, model_dir="saved_models", frames_per_second=1):
    """
    Generate caption for a single video.
    """

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return None

    model_path = os.path.join(model_dir, "best_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.keras")

    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    config_path = os.path.join(model_dir, "config.pkl")

    generator = CaptionGenerator(model_path, tokenizer_path, config_path)
    return generator.generate_caption_for_video(video_path, frames_per_second)


def generate_captions_batch(videos_dir, model_dir="saved_models"):
    """
    Generate captions for all videos in a directory.
    """

    video_files = [
        f for f in os.listdir(videos_dir)
        if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print("No videos found")
        return

    model_path = os.path.join(model_dir, "best_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.keras")

    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    config_path = os.path.join(model_dir, "config.pkl")

    generator = CaptionGenerator(model_path, tokenizer_path, config_path)

    for video in video_files:
        print(f"\nProcessing: {video}")
        caption = generator.generate_caption_for_video(
            os.path.join(videos_dir, video)
        )
        print(f"Caption: {caption}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Video Caption Generator")
    parser.add_argument("--video_path", type=str, help="Path to video file")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--videos_dir", type=str, default="data/videos")
    parser.add_argument("--model_dir", type=str, default="saved_models")
    parser.add_argument("--fps", type=int, default=1)

    args = parser.parse_args()

    print("=" * 60)
    print("VIDEO CAPTION GENERATION")
    print("=" * 60)

    if args.batch:
        generate_captions_batch(args.videos_dir, args.model_dir)

    elif args.video_path:
        caption = generate_caption(
            args.video_path, args.model_dir, args.fps
        )
        print("\nGENERATED CAPTION:")
        print(caption)

    else:
        print("Usage:")
        print("python inference.py --video_path data/videos/demo.mp4")
        print("python inference.py --batch --videos_dir data/videos")
