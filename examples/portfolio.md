# Portfolio Optimization
reference
* http://web.stanford.edu/~boyd/papers/pdf/cvx_applications.pdf
* https://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/applications/portfolio_optimization.ipynb

```julia
# Generate data for long only portfolio optimization.
using LinearAlgebra
using Random
using Convex
using SCS; solver = SCSSolver(verbose=0)

Random.seed!(0)
n = 10
μ = abs.(randn(n))
Σ = (S=randn(n,n); S'*S); @assert isposdef(Σ)
# Long only portfolio optimization.
w = Variable(n)
γ = Constant([0.0])
ret = μ'*w # return
risk = quadform(w, Σ)
prob = maximize(ret-γ*risk, [sum(w)==1, w>=0])
Convex.solve!(prob, solver)
@show prob.status
@show prob.optval
@show w.value

# Compute trade-off curve.
SAMPLES = 100
risk_data = zeros(SAMPLES)
ret_data = zeros(SAMPLES)
γ_vals = exp10.(range(-2,3,length=SAMPLES)) # logspace
for i=1:SAMPLES
    @show γ_vals[i]
    copyto!(γ.value, γ_vals[i])
    Convex.solve!(prob, solver)
    risk_data[i] = sqrt(evaluate(risk))[1]
    ret_data[i] = evaluate(ret)[1]
end
```
```julia
# Plot long only trade-off curve.
using Plots
gr()
markers_on = [29, 40]
fig = plot()
plot(risk_data, ret_data, c=:green, legend=nothing, xlabel="Risk(standard deviation)", ylabel="Return")

# for marker in markers_on:
#     plt.plot(risk_data[marker], ret_data[marker], 'bs')
#     ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
scatter!([sqrt(Σ[i,i]) for i=1:n], μ, c=:red, shape=:c, ms=5)

# for i in range(n):
#     plt.plot(sqrt(Sigma[i,i]).value, mu[i], 'ro')
# plt.xlabel('Standard deviation')
# plt.ylabel('Return')
# plt.show()

```
```julia
# Plot return distributions for two points on the trade-off curve.
using Plots

for (midx, idx) in enumerate(markers_on)
    copyto!(γ.value, γ_vals[idx])
    Convex.solve!(prob, solver)
    x = LinRange(-2, 5, 1000)

#    plot(x, mlab.normpdf(x, ret.value, risk.value), label=r"$\gamma = %.2f$" % gamma.value)
end
plot!(xlabel="Return", ylabel="Density", legend=:topright)
```
