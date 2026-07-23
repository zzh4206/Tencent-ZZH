import numpy as np


ket0 = np.array([1.0, 0.0])
ket1 = np.array([0.0, 1.0])


ket010 = np.kron(np.kron(ket0, ket1), ket0)

ket101 = np.kron(np.kron(ket1, ket0), ket1)

result = (ket010 - ket101) / np.sqrt(2)

print("=" * 60)
print("( |010⟩ - |101⟩ ) / √2  的列向量表示：")
print("=" * 60)
print(result)
print(f"\n维度: {len(result)} (2³ = 8)")
print(f"归一化检查: ||ψ||² = {np.sum(np.abs(result)**2):.6f}")


print("\n各基矢分量：")
for i in range(8):
    bits = f"{i:03b}"
    coef = result[i]
    if abs(coef) > 1e-10:
        print(f"  |{bits}⟩: {coef.real:+.6f}")

print("\n即: (|010⟩ - |101⟩)/√2 = (0, 0, 1/√2, 0, 0, 0, -1/√2, 0)^T")
print("对应 |010⟩ (第2位=1) 和 |101⟩ (第5位=1)，系数分别为 +1/√2 和 -1/√2。")


print("\n" + "=" * 60)
print("Dirac 符号重写小结：")
print("=" * 60)
print("""
2-1: |v'⟩ = R(θ)|0⟩ = cosθ|0⟩ - sinθ|1⟩
2-2: e^{iθσ_x} = cosθ·I + i sinθ·(|0⟩⟨1|+|1⟩⟨0|)
     e^{iθσ_y} = cosθ·I + i sinθ·(-i|0⟩⟨1|+i|1⟩⟨0|)
     e^{iθσ_z} = cosθ·I + i sinθ·(|0⟩⟨0|-|1⟩⟨1|)
2-3: ⟨Q⟩ = ⟨0|e^{-iθ/2 P} Q e^{iθ/2 P}|0⟩
2-4: H|0...0⟩ = n|0...0⟩ + Σ_i |...01_i1_{i+1}...⟩ (flipped pairs)
""")
