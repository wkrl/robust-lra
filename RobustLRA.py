from dataclasses import dataclass
import numpy as np
from typing import Optional, Union


@dataclass
class SketchShapes:
    phi: int
    psi: int
    s: int
    t: int

    @classmethod
    def from_matrix(cls, A: np.ndarray) -> "SketchShapes":
        return cls(
            phi=min(A.shape),
            psi=max(A.shape),
            s=min(A.shape),
            t=max(A.shape),
        )


@dataclass
class RobustLRA:
    A: np.ndarray
    k: Optional[int]
    sketch_shapes: SketchShapes
    eps: Union[float, int]

    def __post_init__(self):
        n, d = self.A.shape

        C_phi = np.log(d)
        C_psi = np.log(n)
        C_s = np.log(d)
        C_t = np.log(n)

        self.Phi = np.random.normal(size=(self.sketch_shapes.phi, n))
        self.Psi = np.random.normal(size=(d, self.sketch_shapes.psi))

        self.T = np.random.normal(size=(d, self.sketch_shapes.t))
        self.S = np.random.normal(size=(self.sketch_shapes.s, n))

        self._N_1 = np.random.laplace(0, C_psi / self.eps, size=(self.sketch_shapes.phi, d))
        self._N_2 = np.random.laplace(0, C_phi / self.eps, size=(n, self.sketch_shapes.psi))
        self._N_3 = np.random.laplace(0, (C_s * C_t) / self.eps, size=(self.sketch_shapes.s, self.sketch_shapes.t))

    def compute(self) -> np.ndarray:
        Y_r = self.Phi @ self.A + self._N_1
        Y_c = self.A @ self.Psi + self._N_2

        Z_r = Y_r @ self.T
        Z_c = self.S @ Y_c
        Z = self.S @ self.A @ self.T + self._N_3

        U_c, s_c, Vh_c = np.linalg.svd(Z_c, full_matrices=False)
        S_c = np.diag(s_c)
        V_c = Vh_c.T

        U_r, s_r, Vh_r = np.linalg.svd(Z_r, full_matrices=False)
        S_r = np.diag(s_r)
        V_r = Vh_r.T

        B = U_c.T @ Z @ V_r

        U_b, s_b, V_b = np.linalg.svd(B)
        Bk = U_b[:, : self.k] @ np.diag(s_b[: self.k]) @ V_b[: self.k, :]

        X_hat = V_c @ np.linalg.pinv(S_c) @ Bk @ np.linalg.pinv(S_r) @ U_r.T

        return Y_c @ X_hat @ Y_r
