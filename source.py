import sympy as sp

x, y = sp.symbols('x y')
rho = 1.0 
k = 1.0

u = (2*x**2 - x**4 - 1) * (y - y**3)
v = -(2*y**2 - y**4 - 1) * (x - x**3)

T_mms = 400 + 50*sp.cos(sp.pi*x) + 100*sp.sin(sp.pi*x*y)

dT_dx = sp.diff(T_mms, x)
dT_dy = sp.diff(T_mms, y)
d2T_dx2 = sp.diff(T_mms, x, 2)
d2T_dy2 = sp.diff(T_mms, y, 2)

S_expr = sp.simplify(rho * (u * dT_dx + v * dT_dy) - k * (d2T_dx2 + d2T_dy2))

print(S_expr)