import numpy as np
import scipy.sparse as sp

def fem_like_banded(n, bw, noise=0.3, dominance=1.05, seed=None):
    rng = np.random.default_rng(seed)
    A = sp.lil_matrix((n, n))

    for i in range(n):
        row_sum = 0.0

        for k in range(1, bw+1):
            if i - k >= 0:
                v = -abs(1.0 + noise * rng.standard_normal())
                A[i, i-k] = v
                A[i-k, i] = v
                row_sum += abs(v)

            if i + k < n:
                v = -abs(1.0 + noise * rng.standard_normal())
                A[i, i+k] = v
                A[i+k, i] = v
                row_sum += abs(v)

        A[i, i] = dominance * row_sum

    b = rng.standard_normal(n)
    print(A)
    return A.tocsr(), b

if __name__ == "__main__":
    fem_like_banded(5, 1)