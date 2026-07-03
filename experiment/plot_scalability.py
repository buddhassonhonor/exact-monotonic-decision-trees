import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

df = pd.read_csv('scalability_5reps_summary.csv')

plt.figure(figsize=(6, 4))
plt.errorbar(df['N'], df['time_mean'], yerr=df['time_std'], fmt='-o', capsize=5, label='EMDT (depth=3)')
plt.xlabel('Sample Size (N)')
plt.ylabel('Runtime (seconds)')
plt.title('Scalability on Linear Dataset (15s limit)')
plt.xticks([100, 200, 300, 400])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('../figures/scalability.png', dpi=300)
print("Saved to ../figures/scalability.png")
