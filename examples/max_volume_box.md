# Maximizing the volume of a box
&copy; Keisuke Uto
* [Maximizing the volume of a box(cvxpy)](https://www.cvxpy.org/examples/dgp/max_volume_box.html)

## original formulation (non-convex)
$$
\begin{aligned}
& \underset{h, w, d \in \mathbb{R}_+}{\text{maximize}} & & hwd \\
& \text{s.t.} & & 2(hw + hd) \le A_{wall}\\
& & & wd \le A_{floor}\\
& & & \alpha \le h/w \le \beta\\
& & & \gamma \le d/w \le \delta\\
\end{aligned}
$$

## convex formulation
$$
\begin{aligned}
& \underset{u_h, u_w, u_d \in \mathbb{R}}{\text{maximize}} & & u_h + u_w + u_d \\
& \text{s.t.} & & \log2 + \log(\exp(u_h+u_w) + \exp(u_h+u_d)) \le \log A_{wall}\\
& & & u_w+u_d \le \log A_{floor}\\
& & & \log\alpha \le u_h-u_w \le \log\beta\\
& & & \log\gamma \le u_d-u_w \le \log\delta\\
\end{aligned}
$$

```julia
# Maximizing the volume of a box
using Convex
using SCS; solver = SCSSolver(verbose=0)

A_wall = 100
A_flr = 10
α, β, γ, δ = 0.5, 2, 0.5, 2

## can't solve GP direcly (cvxpy can solve DGP)
# h = Variable(1, Positive())
# w = Variable(1, Positive())
# d = Variable(1, Positive())
# constr = [
#   2*(h*w + h*d) <= A_wall,
#   w*d <= A_flr,
#   α <= h/w, h/w <= β,
#   γ <= d/w, d/w <= δ,
# ]
# prob = maximize(h*w*d, constr)

u_h = Variable() # h = exp(u_h)
u_w = Variable() # w = exp(u_w)
u_d = Variable() # d = exp(u_d)
constr = [
  log(2) + logsumexp([u_h + u_w, u_h + u_d]) <= log(A_wall),
  u_w + u_d <= log(A_flr),
  log(α) <= u_h-u_w, u_h-u_w <= log(β),
  log(γ) <= u_d-u_w, u_d-u_w <= log(δ),
]
prob = maximize(u_h+u_w+u_d, constr)
Convex.solve!(prob, solver)
@show prob.status
@show prob.optval
# h, w, d = map(x->exp(x.value), [u_h, u_w, u_d])
h, w, d = exp.([u_h.value, u_w.value, u_d.value])
@show h,w,d
```
(h, w, d) = (7.7459648011869335, 3.872979633081028, 2.581992420518033)
