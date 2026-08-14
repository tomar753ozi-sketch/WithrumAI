"""
Whitrum AI - llama.cpp Q8 Runner
Founder: Oguzhan (Dr0xy-Drawn)

Usage:
  python run_model.py --model whitrum-350m-q8_0.gguf
  python run_model.py --model whitrum-350m-q8_0.gguf --interactive
"""

import argparse
import subprocess
import sys
import os


def find_llama_cpp():
    """Find llama-cli or llama-main executable."""
    paths = [
        "./llama-cli",
        "./llama-main",
        "../llama.cpp/build/bin/llama-cli",
        "../llama.cpp/build/bin/llama-main",
        "/usr/local/bin/llama-cli",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def run_inference(model_path, prompt, n_predict=256, temperature=0.7, ctx_size=4096):
    """Run single inference with llama-cli."""
    exe = find_llama_cpp()
    if not exe:
        print("ERROR: llama-cli not found. Build llama.cpp first.")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && make -j")
        sys.exit(1)

    cmd = [
        exe,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-t", "4",
        "--ctx-size", str(ctx_size),
        "--temp", str(temperature),
        "--repeat-penalty", "1.1",
        "--color",
    ]

    print(f"Running: {' '.join(cmd[:5])}...")
    print("-" * 50)
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_interactive(model_path, ctx_size=4096):
    """Run interactive chat mode."""
    exe = find_llama_cpp()
    if not exe:
        print("ERROR: llama-cli not found.")
        sys.exit(1)

    system_prompt = (
        "You are Whitrum AI, created by Oguzhan (Dr0xy-Drawn). "
        "You are a helpful assistant. "
        "You can answer questions in Turkish and English."
    )

    cmd = [
        exe,
        "-m", model_path,
        "-t", "4",
        "--ctx-size", str(ctx_size),
        "--temp", "0.7",
        "--repeat-penalty", "1.1",
        "--color",
        "-i",
        "--system-prompt", system_prompt,
    ]

    print("=" * 50)
    print("Whitrum AI Interactive Mode")
    print("Founder: Oguzhan (Dr0xy-Drawn)")
    print("Type 'exit' to quit")
    print("=" * 50)
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Whitrum AI - llama.cpp Runner")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model file")
    parser.add_argument("--prompt", "-p", default=None, help="Input prompt")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--n-predict", "-n", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--ctx-size", "-c", type=int, default=4096, help="Context window size")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    if args.interactive:
        run_interactive(args.model, args.ctx_size)
    elif args.prompt:
        run_inference(args.model, args.prompt, args.n_predict, args.temperature, args.ctx_size)
    else:
        run_interactive(args.model, args.ctx_size)


if __name__ == "__main__":
    main()
