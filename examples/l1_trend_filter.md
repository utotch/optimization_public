# L1 trend filtering
* [cvxpy sample ref](https://www.cvxpy.org/examples/applications/l1_trend_filter.html)

```julia
fname = dirname(@__FILE__) * "/assets/snp500.txt"
ss = open(fname, "r") do f readlines(f) end
y = parse.(Float64, ss)
```

```julia
using Convex
# using SCS; solver = SCSSolver(verbose=0)
using ECOS; solver = ECOSSolver(verbose=0)

n = length(y)
e1 = ones(n)
D = Array(spdiagm(0=>e1, 1=>-2*e1, 2=>e1)[1:n-2,1:n])
λs = [[10.0], [50.0], [100.0]]
λ_param = Constant([0.0])
x = Variable(n)
prob = minimize(0.5*sumsquares(y-x) + λ_param*norm(D*x,1))
xs = map(λs) do λ
    copyto!(λ_param.value, λ)
    @time Convex.solve!(prob, solver)
    @show prob.status
    @show prob.optval
    x.value
end
# @show x.value
```

```julia
using Plots
fig = plot(y, title="snp500 time series", xlabel="date", ylabel="log price", label="raw data", legend=:bottomright)
for i=1:length(xs)
#    plot!(xs[i], label="l1 trend (\\lambda=$(λs[i]))")
    plot!(xs[i], label="l1 trend (\\lambda=$(λs[i][1]))", lw=2.0)
end
fig
```
