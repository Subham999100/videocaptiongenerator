"""
demo.py - Demonstration and Testing Script

This script demonstrates how each component works with sample data.
Useful for understanding the flow before running on real videos.
"""

import numpy as np
from utils import (create_tokenizer, preprocess_captions, 
                   caption_to_sequence, sequence_to_caption,
                   get_max_caption_length, pad_sequence_to_length,
                   clean_caption)


def demo_tokenization():
    """
    Demonstrate how tokenization works.
    """
    print("="*60)
    print("DEMO 1: TOKENIZATION")
    print("="*60)
    
    # Sample captions
    captions = [
        "<start> a person walking in the park <end>",
        "<start> a dog running on the beach <end>",
        "<start> a cat sitting on a chair <end>",
        "<start> children playing in the playground <end>"
    ]
    
    print("\nOriginal Captions:")
    for i, caption in enumerate(captions, 1):
        print(f"{i}. {caption}")
    
    # Create tokenizer
    print("\n1. Creating tokenizer (building vocabulary)...")
    tokenizer = create_tokenizer(captions)
    
    # Show vocabulary
    print("\n2. Vocabulary (word → number mapping):")
    word_index = tokenizer.word_index
    for word, index in list(word_index.items())[:10]:
        print(f"   '{word}' → {index}")
    print(f"   ... (total {len(word_index)} unique words)")
    
    # Convert caption to sequence
    print("\n3. Converting caption to sequence:")
    test_caption = captions[0]
    sequence = caption_to_sequence(test_caption, tokenizer)
    print(f"   Caption: {test_caption}")
    print(f"   Sequence: {sequence}")
    
    # Convert back to caption
    print("\n4. Converting sequence back to caption:")
    reconstructed = sequence_to_caption(sequence, tokenizer)
    print(f"   Sequence: {sequence}")
    print(f"   Caption: {reconstructed}")
    
    # Clean caption
    print("\n5. Removing start/end tokens:")
    cleaned = clean_caption(reconstructed)
    print(f"   Original: {reconstructed}")
    print(f"   Cleaned: {cleaned}")
    
    print("\n✓ Tokenization demo complete!")
    return tokenizer, captions


def demo_padding():
    """
    Demonstrate sequence padding.
    """
    print("\n" + "="*60)
    print("DEMO 2: PADDING")
    print("="*60)
    
    # Sample sequences of different lengths
    sequences = [
        [2, 5, 12],           # Short sequence
        [2, 5, 12, 34, 7],    # Medium sequence
        [2, 5, 12, 34, 7, 9, 15]  # Long sequence
    ]
    
    print("\nOriginal sequences (different lengths):")
    for i, seq in enumerate(sequences, 1):
        print(f"{i}. Length {len(seq)}: {seq}")
    
    # Find max length
    max_length = max(len(seq) for seq in sequences)
    print(f"\nMaximum length: {max_length}")
    
    # Pad all sequences
    print(f"\nPadding all sequences to length {max_length}:")
    padded_sequences = []
    for i, seq in enumerate(sequences, 1):
        padded = pad_sequence_to_length(seq, max_length)
        print(f"{i}. {seq} → {padded}")
        padded_sequences.append(padded)
    
    print("\n✓ All sequences now have same length!")
    print("✓ Padding demo complete!")
    return padded_sequences


def demo_training_pairs():
    """
    Demonstrate how training pairs are created.
    """
    print("\n" + "="*60)
    print("DEMO 3: TRAINING PAIRS GENERATION")
    print("="*60)
    
    # Sample caption
    caption = "<start> a dog running <end>"
    sequence = [2, 5, 12, 34, 3]  # Corresponding sequence
    
    print(f"\nCaption: {caption}")
    print(f"Sequence: {sequence}")
    print("\nGenerating training pairs (teacher forcing):")
    print("\nFormat: [input_sequence] → target_word")
    print("-" * 60)
    
    # Generate pairs
    for i in range(1, len(sequence)):
        input_seq = sequence[:i]
        target = sequence[i]
        
        # Pad input sequence to fixed length
        padded_input = np.zeros(10)  # max_length = 10
        padded_input[:len(input_seq)] = input_seq
        
        print(f"Step {i}:")
        print(f"  Input:  {list(padded_input.astype(int))}")
        print(f"  Target: {target}")
        print()
    
    print("Explanation:")
    print("- Each step takes partial caption as input")
    print("- Model learns to predict the next word")
    print("- This is called 'teacher forcing'")
    print("\n✓ Training pairs demo complete!")


def demo_model_prediction():
    """
    Demonstrate how model prediction works (conceptual).
    """
    print("\n" + "="*60)
    print("DEMO 4: CAPTION GENERATION (CONCEPTUAL)")
    print("="*60)
    
    print("\nHow the model generates captions step by step:")
    print("-" * 60)
    
    # Simulated vocabulary
    vocab = {
        0: '<pad>',
        1: '<unk>',
        2: '<start>',
        3: '<end>',
        5: 'a',
        12: 'dog',
        34: 'running',
        7: 'person',
        9: 'walking'
    }
    
    # Simulate generation process
    generated_sequence = [2]  # Start with <start>
    max_steps = 5
    
    print("\nStarting generation with <start> token")
    print(f"Current sequence: {generated_sequence} → ['{vocab[2]}']")
    
    # Simulate predictions (in reality, these come from the model)
    simulated_predictions = [5, 12, 34, 3]  # a, dog, running, <end>
    
    for step, next_word in enumerate(simulated_predictions, 1):
        print(f"\nStep {step}:")
        print(f"  Model predicts: {next_word} ('{vocab.get(next_word, 'unknown')}')")
        generated_sequence.append(next_word)
        caption_words = [vocab.get(idx, '?') for idx in generated_sequence]
        print(f"  Current sequence: {generated_sequence}")
        print(f"  Caption so far: {' '.join(caption_words)}")
        
        if next_word == 3:  # <end> token
            print(f"  ✓ Found <end> token, stopping generation")
            break
    
    # Final caption
    final_words = [vocab.get(idx, '?') for idx in generated_sequence 
                   if vocab.get(idx) not in ['<start>', '<end>', '<pad>']]
    final_caption = ' '.join(final_words)
    
    print("\n" + "-" * 60)
    print(f"Final caption: '{final_caption}'")
    print("\n✓ Caption generation demo complete!")


def demo_features_explanation():
    """
    Explain what CNN features represent.
    """
    print("\n" + "="*60)
    print("DEMO 5: UNDERSTANDING CNN FEATURES")
    print("="*60)
    
    # Simulate feature vector
    print("\nCNN (InceptionV3) extracts 2048 features from each frame")
    print("\nExample feature vector (showing first 10 values):")
    sample_features = np.random.randn(2048)
    print(f"  {sample_features[:10]}")
    print(f"  ... (2048 total values)")
    
    print("\nWhat do these numbers mean?")
    print("  - Each number represents presence/absence of certain patterns")
    print("  - Examples of patterns:")
    print("    • Edges and corners")
    print("    • Textures (smooth, rough, striped)")
    print("    • Shapes (round, square, elongated)")
    print("    • Object parts (wheels, faces, fur)")
    print("    • Colors and lighting")
    
    print("\n  - Similar images have similar feature vectors")
    print("  - Example:")
    print("    Image of dog: [0.2, 0.5, ..., 0.1]")
    print("    Image of cat: [0.2, 0.4, ..., 0.2]  (similar values)")
    print("    Image of car: [0.9, 0.1, ..., 0.8]  (different values)")
    
    print("\n  - The LSTM learns to map these features to words")
    print("  - High values in certain positions → predicts 'dog'")
    print("  - High values in other positions → predicts 'car'")
    
    print("\n✓ Feature explanation complete!")


def main():
    """
    Run all demonstrations.
    """
    print("\n" + "="*80)
    print(" "*20 + "VIDEO CAPTION GENERATOR - DEMO")
    print("="*80)
    print("\nThis script demonstrates how each component works.")
    print("No actual videos or models needed - just understanding!")
    print("="*80)
    
    # Run demos
    tokenizer, captions = demo_tokenization()
    input("\nPress Enter to continue to next demo...")
    
    padded_sequences = demo_padding()
    input("\nPress Enter to continue to next demo...")
    
    demo_training_pairs()
    input("\nPress Enter to continue to next demo...")
    
    demo_model_prediction()
    input("\nPress Enter to continue to next demo...")
    
    demo_features_explanation()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nYou've learned:")
    print("1. ✓ How tokenization converts words ↔ numbers")
    print("2. ✓ Why we need padding (fixed-size inputs)")
    print("3. ✓ How training pairs are created (teacher forcing)")
    print("4. ✓ How caption generation works step by step")
    print("5. ✓ What CNN features represent")
    
    print("\nNext steps:")
    print("1. Prepare your video dataset")
    print("2. Run feature_extraction.py")
    print("3. Run train.py")
    print("4. Run inference.py")
    
    print("\n" + "="*80)
    print("Good luck with your project! 🚀")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
