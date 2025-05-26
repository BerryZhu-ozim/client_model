import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# ----------------------------
# Configuration
# ----------------------------
EMB_MODEL_NAME = "jinaai/jina-embeddings-v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="Jina Embedding Service")

# ----------------------------
# Models & Tokenizer Initialization
# ----------------------------
# Load embedding model
print(f"Loading embedding model '{EMB_MODEL_NAME}' on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    EMB_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)
model.eval()
print("Embedding model ready")


# ----------------------------
# Request & Response Schemas
# ----------------------------
class EmbeddingRequest(BaseModel):
    data: list[str]


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


# ----------------------------
# Embedding Utility
# ----------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Batch embed a list of texts with mean pooling and L2 normalization.
    """
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        # mean pooling
        emb = out.last_hidden_state.mean(dim=1)
        # L2 normalize
        emb = emb / emb.norm(dim=1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(req: EmbeddingRequest):
    """
    Receive a list of texts and return their embeddings.
    """
    texts = req.data
    if not texts:
        print("Warning: Empty data list received")
        return EmbeddingResponse(embeddings=[])

    print(f"Embedding {len(texts)} texts")
    try:
        embs = embed_texts(texts)
        embeddings_list = embs.tolist()
        print("Embedding generation successful")
        return EmbeddingResponse(embeddings=embeddings_list)
    except Exception as e:
        print(f"Embedding failed: {e}")
        return EmbeddingResponse(embeddings=[])


# ----------------------------
# Run with:
# uvicorn embedding_service:app --host 0.0.0.0 --port 30003 --reload
# ----------------------------
