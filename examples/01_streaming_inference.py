import axiomlm as ax


def main():
    print("=== AxiomLM Streaming Inference Example ===")

    # Initialize a small test model (In practice, load with ax.load_model)
    config = ax.ModelConfig(
        arch="modern", block_size=256, vocab_size=50304, n_layer=4, n_head=4, n_embd=128
    )
    model = ax.Transformer(config)

    # Initialize the O(1) Paged KV-Cache Engine
    engine = ax.InferenceEngine(model, device="cpu")
    prompt = "def calculate_fibonacci(n):"

    print(f"\nPrompt: {prompt}\n")
    print("Generation Streaming:")
    for token in engine.stream(prompt, max_tokens=50, temperature=0.7):
        print(token, end="", flush=True)
    print("\n\nDone.")


if __name__ == "__main__":
    main()
