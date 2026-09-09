with open("axiomlm/models/transformer.py", "r") as f:
    content = f.read()

content = content.replace("from torch.nn import functional as F", "from torch.nn import functional as F\nfrom ..kernels.ops import fused_cross_entropy")

with open("axiomlm/models/transformer.py", "w") as f:
    f.write(content)
