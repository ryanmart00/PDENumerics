using SparseArrays, PyPlot

"""
    Computes the discretization Au = b of -Δu = f on the unit square with boundary 
    condition u = g to 4th order in h = 1/n (the grid width)
    Optionally setting useadhoclaplace = false, giving the true laplacian Lap_f of f 
    uses that instead of using a 5 point scheme to estimate it.

    We achieve 4th order by using the 9 point Laplacian with error
    -Δ_9 u = -Δu - h^2/12 Δ^2 u + O(h^4)
    Since -Δu = f (for the true solution) we see that the scheme 
    -Δ_9 u = f + h^2/12 * Δf has order O(h^4), even if we use an order h^2 
    scheme to estimate Δf (which is done below)
"""
function assemblePoisson(n, f, g, useadhoclaplace =true, Lap_f = 0)
    h = 1.0/n
    N = (n+1)^2
    x = h* (0:n)
    y = x

    if useadhoclaplace
        Lap_f = adhocLaplace(f, h)
    else 
        Lap_f = (x,y) -> h^2 * Lap_f(x,y)
    end

    umap = reshape(1:N, n+1, n+1)
    A = Tuple{Int64, Int64, Float64}[]  #matrix of the finite difference as (row, col, val) 
                                        #row corresponds to output = forcing
                                        #col corresponds to input = u
    b = zeros(N)

    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j]
            if i == 1 || i == n+1 || j == 1 || j == n+1
                # On the boundary need to set u(x_i, y_j) = g(x_i, y_j)
                push!(A, (row, row, 1.0)) # 1 on the diagonal
                b[row] = g(x[i], y[j]) # value u should be 
            else 
                # In the interior use the 9 point stencil multiplied through by 6 h^2
                # Center
                push!(A, (row, row, 20.0))

                # Diagonals
                push!(A, (row, umap[i-1,j-1], -1.0))
                push!(A, (row, umap[i+1,j-1], -1.0))
                push!(A, (row, umap[i-1,j+1], -1.0))
                push!(A, (row, umap[i+1,j+1], -1.0))

                # Edges
                push!(A, (row, umap[i-1,j], -4.0))
                push!(A, (row, umap[i+1,j], -4.0))
                push!(A, (row, umap[i,j-1], -4.0))
                push!(A, (row, umap[i,j+1], -4.0))

                # Forcing
                # Note that we already multiplied by h^2 so 
                # really the laplace f term has h^4
                b[row] = 6 * h^2 * f(x[i], y[j]) + h^2/2.0 * Lap_f(x[i],y[j])
            end
        end
    end

    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    return A, b, x, y
end

"""
    Computes the 5 point Laplacian of a function f with grid size h, to avoid loss of 
    precision we avoid dividing by h^2
"""
function adhocLaplace(f, h)
    # for h sufficiently small we are losing some precision here by 
    # dividing by h^2 when # we're gonna multiply it by h^2 later 
    # anyway so we simply don't
    return (x,y) -> -4.0 * f(x,y) + f(x-h,y) + f(x+h, y) + f(x, y-h) + f(x,y+h)
end

function plotit(x, y, u, file, ti="")
    # Plotting
    clf()
    contour(x, y, u, 10, colors="k")
    contourf(x, y, u, 10)
    axis("equal")

    title(ti)
    colorbar()
    savefig(file)
    clf()
end

"""
    Performs a grid refinement by successively doubling n on the exact solution 
    given in the notebook
"""
function gridRefinePoisson(nstart, num)
    uexact(x,y) = exp(-(4(x - 0.3)^2 + 9(y - 0.6)^2))
    f(x,y) = uexact(x,y) * (26 - (18y - 10.8)^2 - (8x - 2.4)^2)
    hs=(x->1.0/(nstart*(2^x))).(1:num)
    errors=zeros(num)
    for i in (1:num)
        A, b, x, y = assemblePoisson(nstart*(2^i), f, uexact)

        # Solve + reshape solution into grid array
        u = A \ b
        u = reshape(u, length(x), length(y))

        if i == num
            plotit(x,y,u, "test_plot_poisson.png", 
                   "Plot of -Δu = (26 - (18y - 10.8)^2 - (8x - 2.4)^2) 
                   exp(-(4(x - 0.3)^2 + 9(y - 0.6)^2))")
        end

        # Compute error in max-norm
        u0 = uexact.(x, y')
        errors[i] = maximum(abs.(u - u0))
    end

    # calculate linear regression
    beta = [ones(num) log.(hs)] \ log.(errors)
    line = x->beta[1] + x*beta[2]
    xscale("log")
    yscale("log")
    scatter(hs, errors)
    plot(hs, exp.(line.(log.(hs))))
    xlabel("Grid Width h")
    ylabel("True Error err")
    title("Grid Refinement Error for exp(-(4(x - 0.3)^2 + 9(y - 0.6)^2))")
    regError = (log.(errors) - line.(log.(hs)))
    mean = sum(log.(errors))/num
    meanError = (x->x-mean).(log.(errors))
    R2 = 1 - (regError' * regError)/(meanError' * meanError)
    figtext(0.9, 0.15, "err = $(exp(beta[1])) + x^$(beta[2])", ha="right")
    figtext(0.9, 0.2, "R^2 = $(R2)", ha="right")
    
    show()
    savefig("error_plot_poisson.png")
    return beta, R2

end

function channelflow(L,B,H,n)
    h = 1.0/n
    N = (n+1)^2
    xi = h * (0:n)
    eta = xi
    A = sqrt((L - B)^2/4.0 - H^2)

    umap = reshape(1:N, n+1, n+1)
    M = Tuple{Int64, Int64, Float64}[]  #matrix of the finite difference as (row, col, val) 
                                        #row corresponds to output = forcing
                                        #col corresponds to input = u
    b = zeros(N)
    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j]
            denom = 1.0/(B/2.0 + A*eta[j]) 
            if i == n+1 || j == 1
                # Dirichlet boundary
                push!(M, (row, row, 1.0)) # 1 on the diagonal
                b[row] = 0.0 # value u should be 
            elseif i == 1 
                # left boundary (before the upper boundary so the corner will live here)
                push!(M, (row, umap[i,j], -3.0))
                push!(M, (row, umap[i+1,j], 4.0))
                push!(M, (row, umap[i+2,j], -1.0))
                b[row] = 0.0 #multiply through by 2h
            elseif j == n+1
                # upper boundary (multiply through by 2hH)
                # xi part
                push!(M, (row, umap[i+1,j], -A*xi[i]*denom))
                push!(M, (row, umap[i-1,j], A*xi[i]*denom))
                # eta part
                push!(M, (row, umap[i,j], 3.0))
                push!(M, (row, umap[i,j-1], -4.0))
                push!(M, (row, umap[i,j-2], 1.0))
                b[row] = 0.0
            else 
                # In the interior use the crazy mess we calculated
                # Being a bit more verbose to try to midigate errors
                # Center - contributions from 2nd derivatives
                push!(M, (row, umap[i,j], 2.0/h^2 *((1+A^2/H^2 *xi[i]^2)*denom^2 + 1/H^2)))

                # Diagonals - contributions from mixed derivatives
                push!(M, (row, umap[i+1,j+1], 1.0/h^2/2.0 *A/H^2 *xi[i]*denom))
                push!(M, (row, umap[i+1,j-1], -1.0/h^2/2.0 *A/H^2 *xi[i]*denom))
                push!(M, (row, umap[i-1,j+1], -1.0/h^2/2.0 *A/H^2 *xi[i]*denom))
                push!(M, (row, umap[i-1,j-1], 1.0/h^2/2.0 *A/H^2 *xi[i]*denom))

                # Horizontal edges - xi derivatives
                push!(M, (row, umap[i+1,j], -1.0/h^2*(1.0 + A^2/H^2 * xi[i]^2) * denom^2
                          - 1.0/h *A^2/H^2 * xi[i] * denom^2))
                push!(M, (row, umap[i-1,j], -1.0/h^2*(1.0 + A^2/H^2 * xi[i]^2) * denom^2
                          + 1.0/h *A^2/H^2 * xi[i] * denom^2)) 

                # Vertical edges - eta derivatives
                push!(M, (row, umap[i,j+1], -1.0/h^2/H^2))
                push!(M, (row, umap[i,j-1], -1.0/h^2/H^2))

                # Forcing
                b[row] = 1.0
            end
        end
    end

    M = sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), N, N)

    # The solution reshaped into a square matrix (note we don't change anything here 
    # because functions don't transform under change of variables - we just reinterpret the 
    # grid)
    u = reshape(M \ b, length(xi), length(eta))
    # To get the original coordinates we use the inverse map - note that we can't have 
    # x and y independent - we need them to be matrices of the same size as u
    x = [[i*(B/2.0 + A*j) for j in eta] for i in xi]
    y = [[H*j for j in eta] for i in xi]

    f(i,j) = H*(B/2 + A*eta[j])*u[i,j]

    Q = h^2/4.0 * (f(1,1) + f(1,n+1) + f(n+1,1) + f(n+1,n+1)
                   + 2*sum(f(i,1) for i in (2:n)) + 2*sum(f(i,n+1) for i in (2:n)) 
                   + 2*sum(f(1,j) for j in (2:n)) + 2*sum(f(n+1,j) for j in (2:n)) 
                   + 4*sum(f(i,j) for i in (2:n) for j in (2:n)))

    return Q, x, y, u
end

function gridRefineChannel(L, B, H, nums, ref, p=2)
    Qref, xref, yref, uref = channelflow(L, B, H, ref)

    plotit(xref,yref,uref, "test_plot_channel_$(B).png", 
           "Plot of -Δu = 1 for L=$(L), B=$(B), H=$(H) with
           Q=$(Qref)")
    hs = Float64[]
    errors = Float64[]
    A = sqrt((L - B)^2/4.0 - H^2)

    for n in nums
        Q, x, y, u = channelflow(L,B,H,n)

        # Compute error in max-norm
        #push!(errors,abs(Q - Qref))
        f(i,j) = H*(B/2 + A/H*y[i][j])*(abs(u[i,j] - uref[(i-1)*div(ref, n) + 1, (j-1)*div(ref,n)+1])^p)
        push!(errors, trapezoid(f, (1:n+1), (1:n+1),1.0/n)^(1/p))

                                 
                    
        push!(hs, 1.0/n)
    end

    # calculate linear regression
    beta = [ones(length(nums)) log.(hs)] \ log.(errors)
    line = x->beta[1] + x*beta[2]
    xscale("log")
    yscale("log")
    scatter(hs, errors)
    plot(hs, exp.(line.(log.(hs))))
    xlabel("Grid Width h")
    ylabel("Estimated L1 Error u - u_$(ref)")
    title("Grid Refinement Error for Channel Flow, B=$(B)")
    regError = (log.(errors) - line.(log.(hs)))
    mean = sum(log.(errors))/length(nums)
    meanError = (x->x-mean).(log.(errors))
    R2 = 1 - (regError' * regError)/(meanError' * meanError)
    figtext(0.9, 0.15, "err = $(exp(beta[1])) + x^$(beta[2])", ha="right")
    figtext(0.9, 0.2, "R^2 = $(R2)", ha="right")
    
    show()
    savefig("error_plot_flow_$(B).png")
    return beta, R2

end

function trapezoid(f, xs, ys, h)
    nx = length(xs)
    ny = length(ys)
    return h^2/4.0 * (f(xs[1],ys[1]) + f(xs[1],ys[ny]) + f(xs[nx],ys[1]) + f(xs[nx],ys[ny])
                  + 2*sum(f(xs[i],ys[1]) for i in (2:nx)) + 2*sum(f(xs[i],ys[ny]) for i in (2:nx)) 
                  + 2*sum(f(xs[1],ys[j]) for j in (2:ny)) + 2*sum(f(xs[nx],ys[j]) for j in (2:ny)) 
                  + 4*sum(f(xs[i],ys[j]) for i in (2:nx) for j in (2:ny)))
end
