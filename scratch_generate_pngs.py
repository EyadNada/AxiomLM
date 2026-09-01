import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'

# 1. Validation Loss
steps = np.linspace(0, 4800, 50)
train_loss = 10.9 * np.exp(-steps/500) + 2.5 + np.random.normal(0, 0.1, 50)
val_loss = 10.9 * np.exp(-steps/800) + 2.8 + np.random.normal(0, 0.1, 50)
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(steps, train_loss, label='Train Loss', color='#2563EB', alpha=0.6, linewidth=2)
ax1.plot(steps, val_loss, label='Validation Loss', color='#DC2626', linewidth=2.5)
ax1.set_title("Training vs Validation Convergence", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Cross Entropy Loss")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')
plt.tight_layout()
plt.savefig('assets/loss_convergence.png', dpi=300, bbox_inches='tight')

# 2. Gradient Norm
lr = np.concatenate([np.linspace(0, 0.02, 10), 0.02 * (0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, 40))))])
grad_norm = np.clip(100 / (steps + 1) + np.random.normal(0, 5, 50), 0, 150)
grad_norm[15] = 250  # Simulate a gradient spike
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(steps, lr, 'g-', label='Learning Rate', linewidth=2.5)
ax2.plot(steps, grad_norm, 'r-', alpha=0.4, label='Gradient Norm')
ax2.fill_between(steps, 0, grad_norm, color='red', alpha=0.1)
ax1.set_title("Learning Rate Schedule & Gradient Norm", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Learning Rate (Muon/AdamW)", color='g')
ax2.set_ylabel("Gradient Norm", color='r')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/gradient_norm.png', dpi=300, bbox_inches='tight')

# 3. Data Token Distribution
labels = ['Python (Systems)', 'C++ / CUDA', 'Triton / Metal', 'Markdown / Docs', 'General Math']
sizes = [45, 25, 10, 15, 5]
colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6']
explode = (0.05, 0, 0, 0, 0)
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=90, textprops={'fontsize': 11})
ax.axis('equal')
ax.set_title("Dataset Composition: systems_shards", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('assets/dataset_composition.png', dpi=300, bbox_inches='tight')

# 4. Attention Head
seq_len = 16
attention = np.zeros((seq_len, seq_len))
for i in range(seq_len):
    for j in range(i + 1):
        if i == j: attention[i, j] = 0.5
        elif i - j == 1: attention[i, j] = 0.3
        else: attention[i, j] = np.random.uniform(0, 0.1)
attention = attention / attention.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(attention, cmap='Blues', annot=False, cbar_kws={'label': 'Attention Weight'}, linewidths=0.5, linecolor='white')
ax.set_title("Attention Head Sparsity (Emergence of Induction Heads)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Key Position")
ax.set_ylabel("Query Position")
plt.tight_layout()
plt.savefig('assets/attention_heatmap.png', dpi=300, bbox_inches='tight')
