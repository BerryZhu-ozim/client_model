from huggingface_hub import snapshot_download

custom_dir = "model_local"

model_path = snapshot_download(repo_id="Qwen/Qwen2.5-3B-Instruct", cache_dir=custom_dir)
model_path = snapshot_download(repo_id="Qwen/Qwen3-14B", cache_dir=custom_dir)
model_path = snapshot_download(
    repo_id="jinaai/jina-embeddings-v3", cache_dir=custom_dir
)
model_path = snapshot_download(repo_id="Qwen/Qwen3-0.6B", cache_dir=custom_dir)
