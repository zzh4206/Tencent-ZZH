import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")


print("=" * 60)
print("Case 1: Bell 态 + Z₀Z₁（Z₀Z₁ 的本征态）")
print("=" * 60)

c1 = tc.Circuit(2)
c1.h(0)
c1.cx(0, 1)
exact1 = float(K.real(c1.expectation_ps(z=[0, 1])).numpy())

sv1 = c1.state().numpy().flatten()
probs1 = np.abs(sv1) ** 2

np.random.seed(42)
for N in [100, 1000, 10000, 100000]:
    indices = np.random.choice(4, size=N, p=probs1)
    b0 = indices // 2
    b1 = indices % 2
    zvals = (-1.0) ** (b0 + b1)
    estimate = np.mean(zvals)
    std_error = np.std(zvals) / np.sqrt(N)
    print(f"  N={N:6d}: ⟨Z₀Z₁⟩ = {estimate:.4f} ± {std_error:.6f}")

print("→ 方差 = 0，任意 N 都给出精确 1.0（本征态测量无统计涨落）")
print()


print("=" * 60)
print("Case 2: |+⟩ 态 + Z（非本征态，方差最大）")
print("=" * 60)

c2 = tc.Circuit(1)
c2.h(0)
exact2 = float(K.real(c2.expectation_ps(z=[0])).numpy())
print(f"精确 ⟨Z⟩ = {exact2:.6f} (|+⟩ 中 ⟨Z⟩ = 0)")

sv2 = c2.state().numpy().flatten()
probs2 = np.abs(sv2) ** 2


n_trials = 200
N_list = [10, 30, 100, 300, 1000, 3000, 10000, 30000]

mean_errors = []
std_errors = []

np.random.seed(42)
for N in N_list:
    estimates = np.zeros(n_trials)
    for trial in range(n_trials):
        indices = np.random.choice(2, size=N, p=probs2)

        zvals = 1.0 - 2.0 * indices
        estimates[trial] = np.mean(zvals)

    abs_errors = np.abs(estimates - exact2)
    mean_errors.append(np.mean(abs_errors))
    std_errors.append(np.std(abs_errors))
    print(f"  N={N:6d}: mean|error| = {np.mean(abs_errors):.6f}, "
          f"expected σ/√N = {1.0/np.sqrt(N):.6f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

N_arr = np.array(N_list, dtype=float)


ax1.loglog(N_arr, mean_errors, 'o-', label='Mean |error|', lw=2)

ax1.loglog(N_arr, 1.0 / np.sqrt(N_arr), 'k--', alpha=0.7,
           label=r'$1/\sqrt{N}$ scaling')
ax1.set_xlabel('Number of measurements N')
ax1.set_ylabel('Mean absolute error')
ax1.set_title('Error vs N (Bell state, Z measurement)')
ax1.legend()
ax1.grid(True, alpha=0.3)


scaled = np.array(mean_errors) * np.sqrt(N_arr)
ax2.semilogx(N_arr, scaled, 'o-', lw=2)
ax2.axhline(y=np.mean(scaled), color='red', linestyle='--',
            label=f'Const ≈ {np.mean(scaled):.3f}')
ax2.set_xlabel('Number of measurements N')
ax2.set_ylabel(r'|error| × √N')
ax2.set_title('Scaled error (should be ≈ constant)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(r'$\sigma_{\langle O \rangle} \propto 1/\sqrt{N}$ (Standard Quantum Limit)',
             fontsize=14)
plt.tight_layout()
plt.savefig("4-3_测量不确定度.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 60)
print("理论分析：标准量子极限 (SQL)")
print("=" * 60)
print("""
量子测量的统计误差:

1. 单次测量方差：
   Var(Ô) = ⟨Ô²⟩ - ⟨Ô⟩²
   对泡利算符 P (P² = I): Var(P) = 1 - ⟨P⟩² ∈ [0, 1]
   - 本征态: ⟨P⟩ = ±1 → Var = 0（确定性测量）
   - 均匀叠加: ⟨P⟩ = 0 → Var = 1（最大方差）

2. N 次独立测量均值的方差：
   Var(⟨Ô⟩_est) = Var(Ô) / N
   σ(⟨Ô⟩_est) = √(Var(Ô)) / √N

3. 标度律：σ ∝ 1/√N
   这是"标准量子极限"(Standard Quantum Limit, SQL)。
   源自独立测量的中心极限定理，也是经典统计的基本性质。

4. Bell 态 + Z₀Z₁：
   由于 Bell 态是 Z₀Z₁ 的本征态，每次测量结果完全相同，
   Var(Z₀Z₁) = 0，因此误差恒为 0。

5. 超越 SQL：
   量子计量学中，利用纠缠可使精度达到 1/N（海森堡极限），
   超越了独立测量的 1/√N 极限。但那是通过特殊制备的纠缠探针态实现的。
""")
