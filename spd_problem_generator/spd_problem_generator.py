import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path

def generate_spd_banded_problem(n: int, bandwidth: int, seed: int = None, debug: bool = False):
    if seed is not None:
        np.random.seed(seed)

    A = np.zeros((n, n), dtype=np.float32)

    # symmetric band
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            if i <= j:
                val = np.random.uniform(-1.0, 1.0)
                A[i, j] = val
                A[j, i] = val

    # strict diagonal dominance → SPD
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(0.5, 1.0)

    # normalize rows
    row_sums = np.sum(np.abs(A), axis=1)
    A /= np.max(row_sums)

    x = np.random.uniform(-1.0, 1.0, size=n)
    b = A @ x

    if debug:
        np.set_printoptions(precision=4, suppress=True)

        print("\n===== DEBUG OUTPUT =====")
        print(f"n = {n}")
        print(f"bandwidth = {bandwidth}")
        print(f"nnz (dense count) = {np.count_nonzero(A)}")

        if n > 10:
            print("\nMatrix A (upper 10x10 block):")
            print(A[:10, :10])
            print("...\n")

            print("Vector x (first 10 entries):")
            print(x[:10])
            print("...\n")

            print("Vector b (first 10 entries):")
            print(b[:10])
            print("...\n")
        else:
            print("\nMatrix A:")
            print(A)

            print("\nVector x:")
            print(x)

            print("\nVector b:")
            print(b)

        print("========================\n")

    return sp.csr_matrix(A), x, b


def save_problem_binary(base_dir: Path, A_csr: sp.csr_matrix, x: np.ndarray, b: np.ndarray):
    os.makedirs(base_dir, exist_ok=True)

    A_csr = A_csr.tocsr()

    n = A_csr.shape[0]
    nnz = A_csr.nnz
    sparsity = nnz / (n * n) * 100.0

    filename = f"n{n}_sp{sparsity:.2f}.bin"
    filepath = os.path.join(base_dir, filename)

    with open(filepath, "wb") as f:
        np.uint32(n).tofile(f)
        np.uint32(nnz).tofile(f)

        A_csr.indptr.astype(np.uint32).tofile(f)
        A_csr.indices.astype(np.uint32).tofile(f)
        A_csr.data.astype(np.float32).tofile(f)

        x.astype(np.float32).tofile(f)
        b.astype(np.float32).tofile(f)

    print(f"Saved: {filepath}")
    print(f"  n = {n}")
    print(f"  nnz = {nnz}")
    print(f"  sparsity = {sparsity:.2f}%")


if __name__ == "__main__":
    problem_dir = Path(__file__).parent / "test_problems"
    A, x, b = generate_spd_banded_problem(2000, 4, 24, True)
    save_problem_binary(problem_dir, A, x, b)