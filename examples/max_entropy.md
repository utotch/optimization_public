# Entropy Maximization
* [ref](https://www.cvxpy.org/examples/applications/max_entropy.html)

```julia
using Convex
using SCS; solver = SCSSolver(verbose=10)
using Random

Random.seed!(0)
n, m, p = 20, 10, 5
tmp = rand(n)
A = randn(m, n)
b = A*tmp
F = randn(p, n)
g = F*tmp + rand(p)

x = Variable(n)
prob = maximize(entropy(x), [A*x == b, F*x <= g])
@time solve!(prob, solver)
@show prob.status
@show prob.optval
@show x.value
```
