from huggingface_hub import snapshot_download

local_dir = "satellite_to_street_dataset"
snapshot_download(
    repo_id="amanshakesbeer/satellite-to-street",
    repo_type="dataset",
    local_dir=local_dir
)