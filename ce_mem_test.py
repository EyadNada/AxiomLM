import torch
import torch.nn.functional as F
device = torch.device('mps')
logits = torch.randn(4 * 1024, 50000, device=device, requires_grad=True)
targets = torch.randint(0, 50000, (4 * 1024,), device=device)

# Force allocation
torch.mps.empty_cache()
mem1 = torch.mps.current_allocated_memory()
loss = F.cross_entropy(logits, targets)
mem2 = torch.mps.current_allocated_memory()
loss.backward()
mem3 = torch.mps.current_allocated_memory()

print(f"Before CE: {mem1 / 1e6} MB")
print(f"After CE Forward: {mem2 / 1e6} MB (Diff: {(mem2-mem1)/1e6} MB)")
print(f"After CE Backward: {mem3 / 1e6} MB (Diff: {(mem3-mem2)/1e6} MB)")
