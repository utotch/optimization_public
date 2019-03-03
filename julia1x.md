# Use Julia 1.x withou crying

# eye
```julia
using LinearAlgebra
eye(n) = Matrix{Float64}(I,n,n)   # Dense eye
speye(n) = Diagonal{Float64}(I,n) # Sparse eye
```

# sparse
```julia
using SparseArrays # necessary
A = sparse([1 2; 3 4])  # dense -> sparse
B = Array(A)            # sparse -> dense
```

# linsp
ace
```julia
LinRange(10, 20, 5) # linspace(10, 20, 5)
```
