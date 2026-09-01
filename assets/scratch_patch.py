import re

with open('README.md', 'r') as f:
    content = f.read()

new_section = """
## Training Telemetry & Model Health

Here is a visual breakdown of the model's learning process and dataset, designed to be easy to understand:

### 1. Training vs Validation Convergence
![Training Convergence](assets/loss_convergence.png)
> **What this means:** This shows the model is genuinely learning over time, not just memorizing. As both lines go down, the model gets smarter at understanding code and text without overfitting.

### 2. Learning Speed & Stability (Gradient Norm)
![Learning Rate & Gradient Norm](assets/gradient_norm.png)
> **What this means:** This tracks the "speed" at which the model learns (green line) versus how surprised it is by new data (red spikes). Keeping these balanced ensures the training doesn't suddenly crash.

### 3. Training Diet (Dataset Composition)
![Dataset Composition](assets/dataset_composition.png)
> **What this means:** This breaks down exactly what information the model is fed. A heavy focus on Python and C++ ensures the model becomes a specialized expert at generating systems code.

### 4. How the Model "Pays Attention" (Attention Heatmap)
![Attention Heatmap](assets/attention_heatmap.png)
> **What this means:** This visualizes how the model connects different words together. Over time, it learns to "look back" at specific past words (like variable names) to perfectly predict what to type next!

---
"""

# Insert right before '## Architectural Specifications'
content = content.replace("## Architectural Specifications", new_section + "\n## Architectural Specifications")

with open('README.md', 'w') as f:
    f.write(content)
print("README updated.")
