import os
from huggingface_hub import snapshot_download, login

# Get the token from environment variable
hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("Hugging Face token not found. Set HF_TOKEN as an environment variable.")

# Log in using the token
login(token=hf_token)

# Define the local directory where you want to store the model
local_dir = "locuslab/tofu_ft_llama2-7b"

# Download the model and store it in the specified directory
snapshot_download(repo_id="locuslab/tofu_ft_llama2-7b", local_dir=local_dir, local_dir_use_symlinks=False, token=hf_token)

print(f"Model downloaded to {local_dir}")