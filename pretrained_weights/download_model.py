from huggingface_hub import snapshot_download

# 如果你在国内 / 受限网络，先设环境变量，或在脚本里指定 endpoint
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 仓库 ID — BAAI/DIVA
repo_id = "BAAI/DIVA"

# 保存到本地目录
#local_dir = "./pretrained_weights/CLIP"
local_dir = "./pretrained_weights/SD"
snapshot_download(
    repo_id=repo_id,
    repo_type="model",       # 或者 None
    local_dir=local_dir,
    resume_download=True,    # 如果之前中断，可以继续
    # allow_patterns / ignore_patterns 可用于过滤文件，例如只下载 weights、configs 等
    # allow_patterns=["*.py", "*.json", "*.ckpt"],  
)

print("Download complete. Files are in:", local_dir)
