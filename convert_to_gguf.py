import os, subprocess, json
from huggingface_hub import snapshot_download, HfApi

CKPT_REPO = os.environ["HF_CKPT_REPO"]

def main():
    print("下载最终模型权重...")
    snapshot_download(repo_id=CKPT_REPO, local_dir="final_model", allow_patterns=["*.safetensors", "*.json"])
    
    tok = json.load(open("final_model/tokenizer.json"))
    config = {
        "architectures": ["GPT2LMHeadModel"], "model_type": "gpt2",
        "vocab_size": tok["vocab_size"], "n_positions": 256,
        "n_embd": 768, "n_layer": 8, "n_head": 8, "n_inner": 3072,
        "activation_function": "gelu_new", "bos_token_id": 0, "eos_token_id": 0
    }
    with open("final_model/config.json", "w") as f: json.dump(config, f)
    
    subprocess.run("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git", shell=True, check=True)
    subprocess.run("pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt", shell=True, check=True)
    subprocess.run("python llama.cpp/convert_hf_to_gguf.py final_model --outtype f16 --outfile ha-ha-ha-100m.gguf", shell=True, check=True)
    
    print("上传 GGUF 到 HF...")
    HfApi().upload_file(path_or_fileobj="ha-ha-ha-100m.gguf", path_in_repo="ha-ha-ha-100m.gguf", repo_id=CKPT_REPO, repo_type="model")
    print("✅ GGUF 生成并上传完毕")

if __name__ == "__main__":
    main()
