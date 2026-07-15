import numpy as np
from src.text_embedding.encoders.sbert_encoder import SentenceTransformerEncoder

def main():
    print("Initializing SentenceTransformerEncoder...")
    encoder = SentenceTransformerEncoder(model_name="dangvantuan/vietnamese-embedding", device="cpu")
    
    # Test batch short texts
    texts = ["Xin chào Việt Nam", "Đại học Khoa học Tự nhiên"]
    print(f"\n1. Encode Batch (Short texts): {texts}")
    embeddings = encoder.encode_batch(texts)
    print(f"   => Shape: {embeddings.shape}")
    print(f"   => Norm (L2) of first vector: {np.linalg.norm(embeddings[0]):.6f}")
    print(f"   => Norm (L2) of second vector: {np.linalg.norm(embeddings[1]):.6f}")
    
    # Test long text
    long_text = "Đây là một đoạn tóm tắt rất dài. " * 50
    print(f"\n2. Encode Long Text (Chunking & Mean-Pooling) [{len(long_text.split())} words]")
    long_embedding = encoder.encode_long_text(long_text)
    print(f"   => Shape: {long_embedding.shape}")
    print(f"   => Norm (L2): {np.linalg.norm(long_embedding):.6f}")

if __name__ == "__main__":
    main()
