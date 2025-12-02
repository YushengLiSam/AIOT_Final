from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="chitradrishti/AffectNet",
    repo_type="dataset",
    local_dir="./data/AffectNet",
    endpoint="https://hf-mirror.com",
    resume_download=True,
    max_workers=4
)
print("Finish download!")