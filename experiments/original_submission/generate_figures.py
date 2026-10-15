import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json

with open('/home/claude/results.json') as f:
    data = json.load(f)

# ═══════════════════════════════════════════
# Training Curve — Clean, Large, Readable
# ═══════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(6, 3))

ext = data["sanatan_extended_10B"]["metrics"]
val_curve = ext["val_curve"]
steps_v = [p[0] for p in val_curve]
vals_v = [p[1] for p in val_curve]
tokens_v = [s * 32 * 2048 / 1e9 for s in steps_v]

ax.plot(tokens_v, vals_v, color='#2563eb', linewidth=1.8, label='SANATAN (6-path, $p^2$)')
ax.axhline(y=3.27, color='#1e293b', linewidth=1.2, linestyle='--', label='Softmax (3.27)')
ax.axhline(y=4.86, color='#16a34a', linewidth=1, linestyle=':', alpha=0.6, label='4-path $p^4$ @ 1B (4.86)')

ax.set_xlabel('Tokens (Billions)', fontsize=10)
ax.set_ylabel('Validation Loss', fontsize=10)
ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax.set_ylim(3.0, 5.2)
ax.set_xlim(0, max(tokens_v) * 1.02)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=9)

# Annotate gap
ax.annotate('Gap: 31%', xy=(tokens_v[-1], vals_v[-1]), fontsize=8, color='#2563eb',
            xytext=(tokens_v[-1] - 2.5, 3.6),
            arrowprops=dict(arrowstyle='->', color='#2563eb', lw=0.8))

plt.tight_layout()
plt.savefig('/home/claude/fig_training_curve.pdf', bbox_inches='tight')
print("Saved fig_training_curve.pdf")

# ═══════════════════════════════════════════
# Baselines — Clean Horizontal Bar Chart
# ═══════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))

baselines = [
    ("Softmax (3B)", 3.27, '#1e293b'),
    ("SANATAN 6p (10B)", data["sanatan_extended_10B"]["val"], '#2563eb'),
    ("SANATAN 4-path $p^4$ (1B)", data["ablation_4path_power4"]["val"], '#16a34a'),
    ("SANATAN 6p (3B)", 4.85, '#93c5fd'),
    ("Linear Attn (3B)", data["baseline_linear"]["val"], '#f59e0b'),
    ("Performer (3B)", data["baseline_performer_stable"]["val"], '#ef4444'),
    ("cosFormer (3B)", data["baseline_cosformer"]["val"], '#9ca3af'),
]

names = [b[0] for b in baselines]
vals = [b[1] for b in baselines]
colors = [b[2] for b in baselines]

bars = ax.barh(range(len(baselines)), vals, color=colors, edgecolor='white', height=0.55)
ax.set_yticks(range(len(baselines)))
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel('Validation Loss (lower is better)', fontsize=9)
ax.invert_yaxis()
ax.set_xlim(2.5, 9.0)
ax.tick_params(labelsize=8)
ax.grid(True, axis='x', alpha=0.15)

for i, (bar, v) in enumerate(zip(bars, vals)):
    ax.text(v + 0.08, i, f'{v:.2f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/fig_baselines.pdf', bbox_inches='tight')
print("Saved fig_baselines.pdf")

# ═══════════════════════════════════════════
# Ablation — Clean Horizontal Bar Chart
# ═══════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))

ablations = [
    ("4-path, $p^4$ (soft gate)", data["ablation_4path_power4"]["val"], '#16a34a'),
    ("6-path, $p^4$ (hard gate)", data["ablation_power4"]["val"], '#22c55e'),
    ("4-path, $p^2$ (soft gate)", data["ablation_4path"]["val"], '#2563eb'),
    ("3-path (no root)", data["ablation_3path"]["val"], '#f59e0b'),
    ("6-path, no $\\omega$", data["ablation_no_omega"]["val"], '#ef4444'),
    ("6-path, no $\\lambda$", data["ablation_no_rotation"]["val"], '#dc2626'),
    ("6-path, $p^2$ (full)", data["ablation_full_sanatan"]["val"], '#7c3aed'),
]

names = [a[0] for a in ablations]
vals = [a[1] for a in ablations]
colors = [a[2] for a in ablations]

bars = ax.barh(range(len(ablations)), vals, color=colors, edgecolor='white', height=0.55)
ax.set_yticks(range(len(ablations)))
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel('Validation Loss at 1B tokens (lower is better)', fontsize=9)
ax.invert_yaxis()
ax.set_xlim(4.6, 5.7)
ax.tick_params(labelsize=8)
ax.grid(True, axis='x', alpha=0.15)

for i, (bar, v) in enumerate(zip(bars, vals)):
    ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/fig_ablation.pdf', bbox_inches='tight')
print("Saved fig_ablation.pdf")
