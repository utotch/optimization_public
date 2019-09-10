# Hello Convex.jl
https://www.cvxpy.org/

```julia
using Convex
using SCS; solver = SCSSolver(verbose=0)
# using ECOS; solver = ECOSSolver(verbose=0)
using Random

Random.seed!(0)
m = 30
n = 20
A = randn(m, n)
b = randn(m)
x = Variable(n)
cost = sumsquares(A*x-b)
constr = Convex.Constraint[]
constr += [0 <= x, x <= 1]
prob = minimize(cost, constr)
@time Convex.solve!(prob, solver)
@show prob.status
@show prob.optval
@show x.value
@show prob.constraints[1].dual
@show prob.constraints[2].dual
```
