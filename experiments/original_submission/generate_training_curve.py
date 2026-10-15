import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json

with open('/home/claude/results.json') as f:
    data = json.load(f)

fig, ax = plt.subplots(1, 1, figsize=(6, 3))

# Known data points for 6-path p^2 training
tokens_known = [1.0, 3.0]
vals_known = [5.50, 4.85]

# Add detailed curve from extended run
ext = data["sanatan_extended_10B"]["metrics"]
val_curve = ext["val_curve"]
for step, val in val_curve:
    t = step * 32 * 2048 / 1e9
    tokens_known.append(t)
    vals_known.append(val)

pairs = sorted(zip(tokens_known, vals_known))
tokens_plot = [p[0] for p in pairs]
vals_plot = [p[1] for p in pairs]

# Plot the main curve
ax.plot(tokens_plot, vals_plot, color='#2563eb', linewidth=1.8, 
        label='SANATAN (6-path, $p^2$)')

# Softmax reference only
ax.axhline(y=3.27, color='#1e293b', linewidth=1.2, linestyle='--', label='Softmax (3.27)')

# Mark key milestones
ax.plot(1.0, 5.50, 'o', color='#2563eb', markersize=6, zorder=5)
ax.plot(3.0, 4.85, 's', color='#2563eb', markersize=6, zorder=5)
ax.plot(tokens_plot[-1], vals_plot[-1], 'D', color='#2563eb', markersize=6, zorder=5)

# Annotate milestones
ax.annotate('1B: 5.50', xy=(1.0, 5.50), fontsize=7.5,
            xytext=(1.5, 5.65), arrowprops=dict(arrowstyle='->', color='#64748b', lw=0.7),
            color='#64748b')
ax.annotate('3B: 4.85', xy=(3.0, 4.85), fontsize=7.5,
            xytext=(3.8, 5.1), arrowprops=dict(arrowstyle='->', color='#64748b', lw=0.7),
            color='#64748b')
ax.annotate('10B: 4.30', xy=(tokens_plot[-1], vals_plot[-1]), fontsize=7.5,
            xytext=(8.0, 4.55), arrowprops=dict(arrowstyle='->', color='#2563eb', lw=0.7),
            color='#2563eb')

# Gap annotation
ax.annotate('', xy=(10.2, 3.27), xytext=(10.2, 4.30),
            arrowprops=dict(arrowstyle='<->', color='#ef4444', lw=1.2))
ax.text(10.5, 3.78, '31%', fontsize=8, color='#ef4444', fontweight='bold', va='center')

ax.set_xlabel('Tokens (Billions)', fontsize=10)
ax.set_ylabel('Validation Loss', fontsize=10)
ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax.set_ylim(2.9, 5.9)
ax.set_xlim(-0.3, 11.2)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('/home/claude/fig_training_curve.pdf', bbox_inches='tight')
print("Saved")
