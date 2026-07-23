import numpy as np
import tensorcircuit as tc


K = tc.set_backend("tensorflow")


c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)


print("线路结构:")
print(c.draw())


exp_z0z1 = K.real(c.expectation_ps(z=[0, 1]))
print(f"\n⟨Z₀Z₁⟩ (statevector) = {exp_z0z1.numpy():.6f}")


state = c.state()
print(f"\nFinal statevector: {state.numpy()}")
print(f"  = (|00⟩ + |11⟩)/√2 = Bell state |Φ⁺⟩")


ZZ = np.array([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]], dtype=complex)
sv = state.numpy().flatten()
exp_manual = np.real(np.conjugate(sv).T @ (ZZ @ sv))
print(f"\n⟨Z₀Z₁⟩ (manual) = {exp_manual:.6f}")


exp_z0 = K.real(c.expectation_ps(z=[0]))
exp_z1 = K.real(c.expectation_ps(z=[1]))
correlation = K.real(c.expectation_ps(z=[0, 1]))
print(f"\n⟨Z₀⟩ = {exp_z0.numpy():.6f}")
print(f"⟨Z₁⟩ = {exp_z1.numpy():.6f}")
print(f"⟨Z₀Z₁⟩ = {correlation.numpy():.6f}")
print(f"Correlation = ⟨Z₀Z₁⟩ - ⟨Z₀⟩⟨Z₁⟩ = {correlation.numpy() - exp_z0.numpy()*exp_z1.numpy():.6f}")

print("\n" + "=" * 60)
print("结论：⟨Z₀Z₁⟩ = 1")
print("=" * 60)
print("""
Bell 态 |Φ⁺⟩ = (|00⟩+|11⟩)/√2 是 Z₀Z₁ 的 +1 本征态。
这意味着两个 qubit 的 Z 测量结果总是相同：
要么都是 0，要么都是 1（完美正相关）。
""")
