import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")


print("=" * 60)
print("2-1 旋转矩阵（TC 后端）")
print("=" * 60)


def R_tc(theta):
    c = K.cos(theta)
    s = K.sin(theta)
    return K.stack([K.stack([c, s]), K.stack([-s, c])])


v0 = K.convert_to_tensor(np.array([1.0, 0.0]))
theta_test = K.convert_to_tensor(np.pi / 4)
Rmat = R_tc(theta_test)
vp = K.einsum("ij,j->i", Rmat, v0)
print(f"v'({theta_test.numpy():.4f}) = {vp.numpy()}")


print("\n" + "=" * 60)
print("2-2 矩阵指数（TC 后端）")
print("=" * 60)


sx = K.convert_to_tensor(np.array([[0., 1.], [1., 0.]], dtype=np.complex64))
sy = K.convert_to_tensor(np.array([[0., -1.j], [1.j, 0.]], dtype=np.complex64))
sz = K.convert_to_tensor(np.array([[1., 0.], [0., -1.]], dtype=np.complex64))
I_tc = K.convert_to_tensor(np.eye(2, dtype=np.complex64))
PAULI_TC = {"σx": sx, "σy": sy, "σz": sz}


def matrix_exp_tc(P, theta):
    c = K.cast(K.cos(theta), dtype=K.complex64)
    s = K.cast(K.sin(theta), dtype=K.complex64)
    return c * I_tc + K.cast(1j, dtype=K.complex64) * s * P


theta_t = K.convert_to_tensor(np.pi / 3, dtype=K.float32)
for name, P_tc in PAULI_TC.items():
    M = matrix_exp_tc(P_tc, theta_t)
    print(f"{name}: 公式计算结果形状 {M.shape}")


for name, P_tc in PAULI_TC.items():
    M = matrix_exp_tc(P_tc, theta_t)
    M_dag = K.transpose(K.conj(M))
    UUdag = K.einsum("ij,jk->ik", M, M_dag)
    err = K.max(K.abs(UUdag - I_tc)).numpy()
    print(f"  {name} 幺正性误差: {err:.2e}")


print("\n" + "=" * 60)
print("2-3 矩阵期望（TC 线路 API）")
print("=" * 60)


def expectation_tc(p_idx, q_idx, theta):
    c = tc.Circuit(1)

    if p_idx == 0:
        c.rx(0, theta=theta)


    elif p_idx == 1:
        c.ry(0, theta=theta)
    elif p_idx == 2:
        c.rz(0, theta=theta)


    if q_idx == 0:
        return K.real(c.expectation_ps(x=[0]))
    elif q_idx == 1:
        return K.real(c.expectation_ps(y=[0]))
    else:
        return K.real(c.expectation_ps(z=[0]))


thetas_tc = np.linspace(0, 4 * np.pi, 200)
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for p_idx in range(3):
    for q_idx in range(3):
        ax = axes[p_idx, q_idx]
        vals = [expectation_tc(p_idx, q_idx, float(t)) for t in thetas_tc]
        ax.plot(thetas_tc / np.pi, vals)
        ax.set_xlabel(r"$\theta/\pi$")
        ax.set_ylabel(r"$\langle Q \rangle$")
        pn = ["σx", "σy", "σz"][p_idx]
        qn = ["σx", "σy", "σz"][q_idx]
        ax.set_title(f"P={pn}, Q={qn}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

plt.suptitle("2-3 重写: TC 线路 API", fontsize=14)
plt.tight_layout()
plt.savefig("2-6_期望_TC.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 60)
print("2-4 张量积/Ising Hamiltonian（TC 后端）")
print("=" * 60)


def build_Zi(n, i):
    ops = [I_tc] * n
    ops[i] = sz
    result = ops[0]
    for j in range(1, n):
        result = K.einsum("ab,cd->acbd", result, ops[j])
        result = K.reshape(result, (2 ** (j + 1), 2 ** (j + 1)))
    return result


def build_XiXi1(n, i):
    ops = [I_tc] * n
    ops[i] = sx
    ops[i + 1] = sx
    result = ops[0]
    for j in range(1, n):
        result = K.einsum("ab,cd->acbd", result, ops[j])
        result = K.reshape(result, (2 ** (j + 1), 2 ** (j + 1)))
    return result


for n_test in [2, 3]:
    H_tc = K.zeros((2 ** n_test, 2 ** n_test), dtype=K.complex64)
    for i in range(n_test):
        H_tc = H_tc + K.cast(build_Zi(n_test, i), dtype=K.complex64)
    for i in range(n_test - 1):
        H_tc = H_tc + K.cast(build_XiXi1(n_test, i), dtype=K.complex64)


    v0_tc = K.one_hot(0, 2 ** n_test)
    v0_tc = K.cast(v0_tc, dtype=K.complex64)
    expval = K.real(K.einsum("i,ij,j->", K.conj(v0_tc), H_tc, v0_tc))
    print(f"n={n_test}: ⟨H⟩ = {expval.numpy():.6f} (期望 = {n_test})")


print("\n使用 TC 的 quantum.heisenberg 或手动构造：")
print("TC 提供了方便的 API: tc.quantum.heisenberg_hamiltonian()")


for n_test in [2, 3]:
    c = tc.Circuit(n_test)


    z_terms = sum(K.real(c.expectation_ps(z=[i])) for i in range(n_test))
    xx_terms = sum(K.real(c.expectation_ps(x=[i, i + 1])) for i in range(n_test - 1))
    total = z_terms + xx_terms
    print(f"n={n_test}: ⟨ΣZ_i⟩={z_terms.numpy():.4f}, ⟨ΣX_iX_{{i+1}}⟩={xx_terms.numpy():.4f}")
    print(f"         ⟨H⟩ = {total.numpy():.4f} (期望 = {n_test})")

print("\n所有 §2 题目已用 TC 后端重写完毕。")
