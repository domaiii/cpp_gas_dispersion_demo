# advdiff.py — UFL/FFCx form file for (stationary) advection–diffusion with streamline diffusion / SUPG-like stabilization

from basix.ufl import element
import ufl
from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner,
)

# Scalar FE space for u
e = element("Lagrange", "triangle", 1)

# Geometry (2D mesh)
coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, e)

# Vector FE space for beta (wind field)
beta_e = element("Lagrange", "triangle", 1, shape=(2,))
V_beta = FunctionSpace(mesh, beta_e)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
beta = Coefficient(V_beta)

# Physical diffusion coefficient and characteristic length
D_phys = Constant(mesh)
L_char = Constant(mesh)

# Local mesh size
h = ufl.CellDiameter(mesh)

# Characteristic speed |beta|
U_char = ufl.sqrt(inner(beta, beta))

# Peclet number (dimensionless)
Pe = U_char * L_char / D_phys

# SUPG stabilization parameter
tau = 0.5 * h / (4.0 / (Pe * h) + 2.0 * U_char)

# Variational forms
a = (
    D_phys * inner(grad(u), grad(v)) * dx
    + inner(beta, grad(u)) * v * dx
    + tau * inner(beta, grad(u)) * inner(beta, grad(v)) * dx
)

L = (
    f * v * dx
    + tau * f * inner(beta, grad(v)) * dx
)
