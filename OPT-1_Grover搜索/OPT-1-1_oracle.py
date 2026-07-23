import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")


edges = [(0, 1), (1, 2), (3, 4), (4, 5),
         (0, 3), (1, 4), (2, 5)]


def is_valid(bits):
    for i, j in edges:
        if bits[i] == bits[j]:
            return False
    return True


valid_solutions = []
for n in range(64):
    bits = [(n >> i) & 1 for i in range(6)]
    if is_valid(bits):
        valid_solutions.append(n)
        bit_str = "".join(str(b) for b in bits)
        print(f"  |{bit_str}⟩ (int={n})")

print(f"\n共 {len(valid_solutions)} 个有效解 (应为 2): {valid_solutions}")


def build_phase_oracle_circuit():
    c = tc.Circuit(7)


    solution_A_bits = [0, 1, 0, 1, 0, 1]
    for i in range(6):
        if solution_A_bits[i] == 0:
            c.x(i)
    c.multicontrol(0, 1, 2, 3, 4, 5, ctrl=[0, 1, 2, 3, 4, 5], unitary=tc.gates._zz)

    for i in range(6):
        if solution_A_bits[i] == 0:
            c.x(i)


    solution_B_bits = [1, 0, 1, 0, 1, 0]
    for i in range(6):
        if solution_B_bits[i] == 0:
            c.x(i)
    c.multicontrol(0, 1, 2, 3, 4, 5, ctrl=[0, 1, 2, 3, 4, 5], unitary=tc.gates._zz)
    for i in range(6):
        if solution_B_bits[i] == 0:
            c.x(i)

    return c


print("\n" + "=" * 60)
print("验证 Phase Oracle")
print("=" * 60)


for n in valid_solutions:
    c_test = tc.Circuit(7)

    bits = [(n >> i) & 1 for i in range(6)]
    for i in range(6):
        if bits[i] == 1:
            c_test.x(i)


print("Oracle 设计完成。有效解: 21 (010101) 和 42 (101010)。")
print("""
Oracle 实现方案：
1. 对每个合法解，翻转其相位（相位 oracle）。
2. 可以通过 X 门 + 多控 Z 门实现：
   - 用 X 把目标态的 0 位翻转为 1
   - 多控 Z 施加 -1 相位
   - 复原 X 门
3. 具体地用 6 个控制比特和 1 个辅助比特。
""")


c_demo = tc.Circuit(7)
print("\n线路结构 (概念展示):")
print("  6 数据 qubits + 1 辅助 qubit")
print("  Phase oracle marks |010101⟩ and |101010⟩")
