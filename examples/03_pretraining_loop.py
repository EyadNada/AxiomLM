import torch
import axiomlm as ax

def main():
    print("=== AxiomLM Pretraining & Muon Optimizer Example ===")
    
    # 1. Setup Architecture
    config = ax.ModelConfig(
        arch="modern",
        block_size=512,
        vocab_size=50304,
        n_layer=6,
        n_head=8,
        n_embd=512
    )
    model = ax.Transformer(config)
    
    # Auto-detect hardware
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters on {device}.")

    # 2. Configure Muon Optimizer
    # Automatically routes 2D weights to Muon and 1D weights to AdamW
    if hasattr(model, "configure_optimizers"):
        optimizers = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=0.0006,
            muon_lr=0.02,
            device=device,
            optimizer_type="muon"
        )
    else:
        print("Fallback to basic AdamW...")
        optimizers = torch.optim.AdamW(model.parameters(), lr=0.0006)
    
    # 3. Dummy Training Step
    input_ids = torch.randint(0, 50304, (4, 512), device=device)
    targets = torch.randint(0, 50304, (4, 512), device=device)
    
    logits, loss = model(input_ids, targets=targets)
    loss.backward()
    
    print(f"\nForward/Backward pass completed. Initial Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
