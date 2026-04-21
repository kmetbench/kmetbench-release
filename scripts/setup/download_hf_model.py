# Usage: python download_hf_model.py --model_index 18 &
# For loop execution in Linux CLI: for i in {0..56}; do python download_hf_model.py --model_index $i; done &

import time
from huggingface_hub import snapshot_download

MODEL_TAG_LIST = [
    # Cohere models
    "CohereLabs/c4ai-command-a-03-2025",
    "CohereLabs/command-a-reasoning-08-2025",
    # DeepSeek models
    "deepseek-ai/deepseek-vl2",
    "deepseek-ai/deepseek-vl2-small",
    "deepseek-ai/deepseek-vl2-tiny",
    # "deepseek-ai/DeepSeek-R1",
    # "deepseek-ai/deepseek-moe-16b-chat",
    # Gemma models
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    # "meta-llama/Llama-3.2-90B-Vision-Instruct",
    # Microsoft Phi models
    "microsoft/Phi-4",
    "microsoft/Phi-4-mini-instruct",
    "microsoft/Phi-4-mini-reasoning", # 19
    "microsoft/Phi-4-reasoning",
    # InternVL3.5 models
    "OpenGVLab/InternVL3_5-1B",
    "OpenGVLab/InternVL3_5-2B",
    "OpenGVLab/InternVL3_5-4B",
    "OpenGVLab/InternVL3_5-8B",
    "OpenGVLab/InternVL3_5-14B",
    "OpenGVLab/InternVL3_5-38B",
    # InternVL3 models
    "OpenGVLab/InternVL3-1B-Instruct",
    "OpenGVLab/InternVL3-2B-Instruct",
    "OpenGVLab/InternVL3-8B-Instruct",
    "OpenGVLab/InternVL3-9B-Instruct",
    "OpenGVLab/InternVL3-14B-Instruct",
    "OpenGVLab/InternVL3-38B-Instruct",
    "OpenGVLab/InternVL3-78B-Instruct",
    # Qwen models
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",   # 30
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-32B",
    "Qwen/QVQ-72B-Preview",
    # Commented out models
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "LGAI-EXAONE/EXAONE-4.0-32B",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
    "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    "NCSOFT/VARCO-VISION-2.0-1.7B",
    "NCSOFT/VARCO-VISION-2.0-14B",
    "skt/A.X-4.0",
    "skt/A.X-4.0-Light",
    "skt/A.X-4.0-VL-Light", # 50
    "openai/gpt-oss-20b",
    # Meta Llama models
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "openai/gpt-oss-120b", # 56
]


import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def safe_snapshot_download(repo_id, repo_type="model", revision=None, max_retries=5, sleep_seconds=120, timeout_seconds=120, **kwargs):
    """Download a model from Hugging Face Hub with retry mechanism and timeout."""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to download {repo_id}...")
            with timeout(timeout_seconds):
                return snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=revision, **kwargs)
        except TimeoutException:
            print(f"Download timed out after {timeout_seconds} seconds, skipping to next model...")
            raise
        except Exception:
            if attempt < max_retries - 1:
                print(f"Download failed, retrying in {sleep_seconds} seconds...")
                time.sleep(sleep_seconds)
            else:
                print(f"Failed to download {repo_id} after {max_retries} attempts")
                raise


def main(args):
    """Main function to download models starting from the provided index."""
    start_index = args.model_index

    for i in range(start_index, len(MODEL_TAG_LIST)):
        model_tag = MODEL_TAG_LIST[i]
        print(f"Downloading model {i}: {model_tag}")

        try:
            safe_snapshot_download(model_tag)
            print(f"Successfully downloaded: {model_tag}")
            break  # Exit after successful download
        except TimeoutException:
            print(f"Skipping {model_tag} due to timeout, moving to next model...")
            continue
        except Exception as e:
            print(f"Failed to download {model_tag}: {e}")
            print("Moving to next model...")
            continue

    print("Download process completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub.")
    parser.add_argument("--model_index", type=int, required=True, help="Starting index of the model to download from the model tag list.")
    args = parser.parse_args()

    main(args)
