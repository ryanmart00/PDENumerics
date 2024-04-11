"""
    c = mkfdstencil(x, xbar, k)

Compute the coefficients `c` in a finite difference approximation of a function
defined at the grid points `x`, evaluated at `xbar`, of order `k`.
"""
function mkfdstencil(x, xbar, k)
    n = length(x)
    A = @. (x[:]' - xbar) ^ (0:n-1) / factorial(0:n-1)
    b = (1:n) .== k+1
    c = A \ b
end
