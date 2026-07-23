import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")


c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)


exact = float(K.real(c.expectation_ps(z=[0, 1])).numpy())
print(f"精确 ⟨Z₀Z₁⟩ = {exact:.6f}")
print()


for n_shots in [10, 100, 1000, 10000, 100000]:

    bitstrings = []
    for _ in range(n_shots):

        result = c.measure(2)


        pass


    break


np.random.seed(42)


sv = c.state().numpy().flatten()
probs = np.abs(sv) ** 2


print("=" * 60)
print("测量模拟：从 Bell 态采样估计 ⟨Z₀Z₁⟩")
print("=" * 60)

results = {}

for N in [10, 30, 100, 300, 1000, 3000, 10000]:

    indices = np.random.choice(4, size=N, p=probs)


    b0 = indices // 2
    b1 = indices % 2
    z0z1_vals = (-1.0) ** (b0 + b1)
    estimate = np.mean(z0z1_vals)
    std_err = np.std(z0z1_vals) / np.sqrt(N)

    results[N] = (estimate, std_err)
    err = abs(estimate - exact)
    print(f"  N={N:6d}: ⟨Z₀Z₁⟩_est = {estimate:.6f} ± {std_err:.6f}, "
          f"|error| = {err:.6f}")


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
