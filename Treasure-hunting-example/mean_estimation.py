import numpy as np
import matplotlib.pyplot as plt

np.random.seed(402)
mu = 3         # true mean
sigma = 1      # standard var 
N = 2000       # total samples

samples = np.random.normal(mu, sigma, N)

# 1. Batch mean
mean_batch = np.cumsum(samples) / np.arange(1, N+1)

# 2. Robbins-Monro/(α_k = 1/k)
w = samples[0]
w_hist = [w]
for k in range(2, N+1):
    xk = samples[k-1]
    w = w - (1/k)*(w - xk)
    w_hist.append(w)

# 3. Robbins-Monro with perturbation and fixed step (alpha=0.5)
alpha = 0.005
w2 = samples[0]
w2_hist = [w2]
for k in range(2, N+1):
    eta = np.random.randn()
    xk = samples[k-1] + eta
    w2 = w2 - alpha * (w2 - xk)
    w2_hist.append(w2)

plt.figure(figsize=(10, 5))
plt.plot(mean_batch, label='Batch mean (all data)', color='k', linewidth=2)
plt.plot(w_hist, '--', label='Incremental mean $(\\alpha_k=1/k)$', color='b', linewidth=2)
plt.plot(w2_hist, '-.', 
         label=f'Fixed step $(\\alpha={alpha})$ w/ noise', 
         color='r', linewidth=2)
plt.axhline(mu, color='g', linestyle='-', label='True mean', linewidth=3)
plt.xlabel('Sample $k$', fontsize=16)
plt.ylabel('Estimated mean', fontsize=16)
plt.title('Incremental Mean Estimation Comparison (step size experiment)', fontsize=18, pad=12)
plt.legend(fontsize=13, loc='best')
plt.grid(True, linestyle='--', linewidth=1, alpha=0.7)
plt.tight_layout()
plt.savefig("mean_estimation_large_step.png", dpi=120)
plt.close()