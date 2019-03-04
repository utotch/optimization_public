# Optimization in Julia
&copy; Keisuke Uto


I want to share basic convex optimization methods in Julia.
I think these codes includes useful tips for using Convex.jl.

You can render math formula with [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima) if you use Chrome.

## Reimplementation of cvxpy sample
Thanks for useful [examples in cvxpy](https://www.cvxpy.org/examples/index.html).

### Control
* [Control](examples/control.md)

### Regression
* [Ridge Regression](examples/ridge_regression.md)
* [Lasso Regression](examples/lasso_regression.md)
* [Huber Regression](examples/huber_regression.md)

### Dimension Reduction
* [Nonnegative Matrix Factorization](examples/nmf.md)

### Geometric Programming
* [Maximizing the volume of a box](max_volume_box.md)

## Julia tips
* [Julia 1.x without crying](julia1x.md)

## reference
* [cvxpy](https://www.cvxpy.org/)
* [Convex.jl](https://github.com/JuliaOpt/Convex.jl)
  * [convex operations in Convex.jl](https://convexjl.readthedocs.io/en/latest/operations.html)

if you find mistakes, please tell me in [issue](https://github.com/utotch/optimization_public/issues).
