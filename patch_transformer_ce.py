import re

with open("axiomlm/models/transformer.py", "r") as f:
    content = f.read()

# Add import
content = re.sub(r'import torch.nn.functional as F', 
                 'import torch.nn.functional as F\nfrom ..kernels.ops import fused_cross_entropy', 
                 content)

# Replace F.cross_entropy
content = re.sub(r'loss = F\.cross_entropy\(logits\.view\(-1, logits\.size\(-1\)\), targets\.view\(-1\), ignore_index=-1\)',
                 'loss = fused_cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)',
                 content)

with open("axiomlm/models/transformer.py", "w") as f:
    f.write(content)
