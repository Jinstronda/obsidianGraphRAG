#!/usr/bin/env python3
"""
Test script for cross-encoder reranking model
Verifies Windows compatibility and RTX 3060 performance
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cross_encoder_reranker():
    """Test the cross-encoder/ms-marco-MiniLM-L-6-v2 model"""
    
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    try:
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Using device: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            logger.info(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        
        # Test reranking with sample data
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "Python is a programming language commonly used for data science and machine learning applications.",
            "The weather today is sunny with a temperature of 75 degrees Fahrenheit."
        ]
        
        logger.info("Testing reranking performance...")
        start_time = time.time()
        
        # Prepare inputs for cross-encoder
        pairs = [(query, passage) for passage in passages]
        
        # Tokenize and process
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get scores
        with torch.no_grad():
            scores = model(**inputs).logits.squeeze(-1)
        
        # Sort by relevance
        sorted_indices = torch.argsort(scores, descending=True)
        
        inference_time = time.time() - start_time
        logger.info(f"Reranking completed in {inference_time:.3f} seconds")
        
        # Show results
        logger.info("\nReranking Results:")
        for i, idx in enumerate(sorted_indices):
            score = scores[idx].item()
            logger.info(f"{i+1}. Score: {score:.3f} - {passages[idx][:100]}...")
        
        if torch.cuda.is_available():
            logger.info(f"VRAM used after inference: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        
        logger.info("\n✅ Cross-encoder model test PASSED!")
        logger.info("Model is working correctly on your system.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing Cross-Encoder Reranking Model")
    print("Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("="*60)
    
    success = test_cross_encoder_reranker()
    
    if success:
        print("\n🎉 SUCCESS: Your system is ready for the new reranking model!")
        print("You can now use the Graph RAG system without flash-attn issues.")
    else:
        print("\n❌ FAILED: There may be an issue with your setup.")
        print("Please check the error messages above.") 