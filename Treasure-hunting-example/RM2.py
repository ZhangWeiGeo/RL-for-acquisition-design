import numpy as np
import matplotlib.pyplot as plt

g = lambda w: np.tanh(w - 1)
true_root = 1
n_iter = 100

w_hist = []
w = 3.0
for k in range(1, n_iter+1):
    eta = np.random.randn();
    gk = g(w) + eta;
    a_k = 1.0 / k;
    w = w - a_k * gk;
    w_hist.append(w);

plt.figure(figsize=(10, 5))
plt.plot(w_hist, label='RM estimate $w_k$', color='b')
plt.axhline(true_root, color='red', linestyle='--', label=r'True root $w^*=1$')
plt.xlabel('Iteration $k$', fontsize=16)
plt.ylabel('$w_k$', fontsize=16)
plt.title(r'Robbins-Monro on $g(w)=\tanh(w-1)$, $w_1=3$, $a_k=1/k$', fontsize=14)
plt.legend()
plt.ylim(0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig("RM2.png", dpi=120);
plt.close()