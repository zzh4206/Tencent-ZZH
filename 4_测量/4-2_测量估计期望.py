"""
星火预习题目 — 4-2：基于测量结果近似期望
============================================
用测量比特串来估计 ⟨Z₀Z₁⟩。

方法：对线路进行 N 次测量，每次得到比特串 b₀b₁。
对于每个测量结果，Z₀Z₁ 的本征值为 (-1)^{b₀+b₁}
（因为 Z|0⟩=+|0⟩, Z|1⟩=-|1⟩）。

期望估计：
  ⟨Z₀Z₁⟩ ≈ (1/N) Σ_{样本} (-1)^{b₀+b₁}

由于 Bell 态中 b₀=b₁ 总是成立，每个样本都给出 +1，
所以 ⟨Z₀Z₁⟩_est = 1（精确）。
"""

import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")

# Build the circuit
c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

# Exact expectation
exact = float(K.real(c.expectation_ps(z=[0, 1])).numpy())
print(f"精确 ⟨Z₀Z₁⟩ = {exact:.6f}")
print()

# Perform measurements
for n_shots in [10, 100, 1000, 10000, 100000]:
    # Do n_shots measurements
    bitstrings = []
    for _ in range(n_shots):
        # Measure and get bitstring as integer
        result = c.measure(2)  # returns (int, count) or a list
        # Actually, use measure_jax or just sample
        # For simplicity, we use the per-shot API
        pass

    # Better approach: use c.sample() or batch measurement
    # Let's use the proper API
    break

# Actually, let's use the correct API for multiple shots
np.random.seed(42)

# Method: Use the statevector to simulate measurements
sv = c.state().numpy().flatten()
probs = np.abs(sv) ** 2
# Bell state: prob(|00⟩) = 0.5, prob(|11⟩) = 0.5

print("=" * 60)
print("测量模拟：从 Bell 态采样估计 ⟨Z₀Z₁⟩")
print("=" * 60)

results = {}

for N in [10, 30, 100, 300, 1000, 3000, 10000]:
    # Sample bitstrings from the statevector
    indices = np.random.choice(4, size=N, p=probs)
    # For each sample, compute Z₀Z₁ eigenvalue = (-1)^{b₀+b₁}
    # b₀ = index // 2, b₁ = index % 2
    b0 = indices // 2
    b1 = indices % 2
    z0z1_vals = (-1.0) ** (b0 + b1)
    estimate = np.mean(z0z1_vals)
    std_err = np.std(z0z1_vals) / np.sqrt(N)  # Standard error of mean

    results[N] = (estimate, std_err)
    err = abs(estimate - exact)
    print(f"  N={N:6d}: ⟨Z₀Z₁⟩_est = {estimate:.6f} ± {std_err:.6f}, "
          f"|error| = {err:.6f}")

# Visualization
Ns = sorted(results.keys())
estimates = [results[n][0] for n in Ns]
errors = [results[n][1] for n in Ns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.errorbar(Ns, estimates, yerr=errors, fmt='o-', capsize=3)
ax1.axhline(y=exact, color='red', linestyle='--', label=f'Exact = {exact}')
ax1.set_xscale('log')
ax1.set_xlabel('Number of shots N')
ax1.set_ylabel(r'$\langle Z_0 Z_1 \rangle$ estimate')
ax1.set_title('Expectation estimate vs N')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.errorbar(Ns, [abs(e - exact) for e in estimates], yerr=errors, fmt='o-', capsize=3)
ax2.set_xscale('log')
ax2.set_yscale('log')
# 1/√N reference line
ax2.plot(Ns, [1/np.sqrt(n) for n in Ns], 'k--', alpha=0.5, label=r'$1/\sqrt{N}$')
ax2.set_xlabel('Number of shots N')
ax2.set_ylabel('Absolute error')
ax2.set_title('Error vs N')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("4-2_测量估计.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("讨论：")
print("=" * 60)
print("""
1. Bell 态 |Φ⁺⟩ 的测量结果只有 |00⟩ 和 |11⟩，概率各 50%。

2. 对于每次测量，Z₀Z₁ = (-1)^{b₀+b₁}:
   - |00⟩ → b₀=0, b₁=0 → (+1)(+1) = +1
   - |11⟩ → b₀=1, b₁=1 → (-1)(-1) = +1
   所以每 shot 都给出 +1，估计永远是 1。

3. 这是 Z₀Z₁ 在 Bell 态下为 +1 本征态的直接体现。
   测量的比特串永远满足 b₀=b₁，两个 qubit 完美关联。
""")
