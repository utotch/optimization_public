# Perron-Frobenius matrix completion (Pending)
&copy; Keisuke Uto

Convex.jl can't solve lambdamax(asymmetric_matrix). argument are restricted to be symmetric matrix(?).

[cvxpy pf_eigenvalue](https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/atoms/pf_eigenvalue.py) can treat elementwise positive matrix.

## original formulation (non-convex)
$$
\begin{aligned}
& \underset{X \in \mathbb{R_+^{n\times n}}}{\text{minimize}} & & λ_{pf}(X) \\
& \text{s.t.} & & \prod_{(i,j)\notin \Omega} X_{ij} = 1\\
& & & X_{ij} = A_{ij}, (i,j) \in \Omega\\
\end{aligned}
$$

## convex formulation
$$
\begin{aligned}
& \underset{U \in \mathbb{R^{n\times n}}}{\text{minimize}} & & λ_{pf}(X) \\
& \text{s.t.} & & \sum_{(i,j)\notin \Omega} U_{ij} = 0\\
& & & U_{ij} = \log A_{ij}, (i,j) \in \Omega\\
\end{aligned}
$$


```julia
# Debugging: Result is worse (grater) than cvxpy
using Convex
using SCS; solver = SCSSolver(verbose=0)

n = 3
A = [
  1.0 NaN 1.9
  NaN 0.8 NaN
  3.2 5.9 NaN
]

# X = Variable(n,n, Positive())
# prob = minimize(pf_eigenvalue(X), [prod(X[.!isnan.(A)]), X[isnan.(A)].==A[isnan.(A)])
U = Variable(n,n)
# sum(X .* float.(.!isnan.(A)))
constr = [sum([U[i,j] for (i,j) in Tuple.(findall(isnan.(A)))]) == 0]
constr += [U[i,j]==log.(A[i,j]) for (i,j) in Tuple.(findall(.!isnan.(A)))]
prob = minimize(lambdamax(exp(U)), constr) # pf_eigenvalue = lambdamax
Convex.solve!(prob, solver)
@show prob.status
@show prob.optval
@show U.value
X=exp.(U.value)
show(IOContext(stdout), "text/plain", X) # result is wrong because it is restricted to symmetric
println()
# @show X[1,2] * X[2,1] * X[2,3] * X[3,3]

X_true = [1.         4.63616907 1.9
          0.49991744 0.8        0.37774148
          3.2        5.9        1.14221476]
# [Debugging] Result is worse than cvxpy
```
