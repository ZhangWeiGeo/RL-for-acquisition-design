import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

g = lambda w: w**5 - 5
true_root = 5 ** (1/5)
n_iter = 100
w_hist = []

w = 0.0
for k in range(5, n_iter+1):
    eta = np.random.randn();       # random noise
    g_tilde = g(w) + eta;
    a_k = 1 / k;                 # step
    w = w - a_k * g_tilde;
    # w = np.clip(w, -10, 10)       # 
    w_hist.append(w);

plt.figure(figsize=(10, 5))
plt.plot(w_hist, label='RM estimate $w_k$', color='b')
plt.axhline(true_root, color='red', linestyle='--', label=r'True root $5^{1/3} \approx 1.71$')
plt.xlabel('Iteration $k$', fontsize=16)
plt.ylabel('$w_k$', fontsize=16)
plt.title('Robbins-Monro on $g(w) = w^5 - 5$ with noisy observation', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("RM1.png", dpi=120);
plt.close()