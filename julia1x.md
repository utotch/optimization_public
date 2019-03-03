# Use Julia 1.x without crying
&copy; Keisuke Uto

# eye
```julia
using LinearAlgebra
eye(n) = Matrix{Float64}(I,n,n)   # Dense eye
speye(n) = Diagonal{Float64}(I,n) # Sparse eye
```

# srand
```julia
using Random
Random.seed!(0)
```

# sparse
```julia
using SparseArrays # necessary
A = sparse([1 2; 3 4])  # dense -> sparse
B = Array(A)            # sparse -> dense
```

# linspace
```julia
LinRange(10, 20, 5) # linspace(10, 20, 5)
```

# blkdiag (for dense matrix)
```julia
using SparseArrays
blkdiag_(As...) = Array(blockdiag(map(sparse, As)...))
# blockdiag([1 2; 3 4], ones(3,3)) for dense
blkdiag_([1 2; 3 4], ones(3,3))
```
