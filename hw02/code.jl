using LinearAlgebra, SparseArrays, PyPlot, PyCall

function euler_fluxes(r,ru,rv,rE, gamma=7.0/5.0)
    p = (gamma - 1)*(rE - 0.5*(ru.*ru./r + rv.*rv./r))
    return ru, rv, ru.*ru./r + p, ru.*rv./r, rv.*ru./r, rv.*rv./r + p, 
        ru./r .*(rE + p), rv./r.*(rE + p)
end

"""
    Computes a compact divergence on [Fx, Fy] with scale h in 2D by applying to the 
    x and y directions independently. This is achieved by calculating the normal 
    matrices and then computing the kronecker product: a representation of the tensor 
    product in the usual x first then y ordered basis

    We have optional arguments for the size of x and y grid sizes to be different. Specifically 
    (x size) * ymul = (y size) * xmul, so that if we want twice as many xs as ys set 
    xmul/ymul = 2
"""
function compact_div(Fx, Fy, h; xmul::Int64=1, ymul::Int64=1)
    xlen = isqrt(div(size(Fx)[1]*xmul, ymul))
    ylen = div(size(Fx)[1], xlen)
    xI = sparse(Matrix(1.0I,xlen,xlen))
    yI = sparse(Matrix(1.0I,ylen,ylen))
    xLHS, xRHS = compact_div_matrices(xlen)
    yLHS, yRHS = compact_div_matrices(ylen)
    return (kron(yI, xLHS) \ (kron(yI, xRHS) * Fx) 
            + kron(yLHS, xI) \ (kron(yRHS, xI) * Fy))/h
end

const _div_memo = Dict()
function compact_div_matrices(len::Int64)
    if len in keys(_div_memo)
        return _div_memo[len]
    end
    ext = [len; 1:len ; 1]
    umap(i) = ext[i+1]

    LHS = Tuple{Int64, Int64, Float64}[] 
    RHS = Tuple{Int64, Int64, Float64}[] 
    for i in (1:len)
        row = umap(i)
        push!(LHS, (row, umap(i-1), 1.0))
        push!(LHS, (row, umap(i), 4.0))
        push!(LHS, (row, umap(i+1), 1.0))

        push!(RHS, (row, umap(i+1), 3.0))
        push!(RHS, (row, umap(i-1), -3.0))
    end
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), len, len)
    RHS = sparse((x->x[1]).(RHS), (x->x[2]).(RHS), (x->x[3]).(RHS), len, len) 
    _div_memo[len] = (LHS,RHS)
    return LHS, RHS
end

"""
    Computes a compact filter on u with parameter alpha in 2D by applying to the 
    x and y directions independently. This is achieved by calculating the normal 
    matrices and then computing the kronecker product: a representation of the tensor 
    product in the usual x first then y ordered basis

    We have optional arguments for the size of x and y grid sizes to be different. Specifically 
    (x size) * ymul = (y size) * xmul, so that if we want twice as many xs as ys set 
    xmul/ymul = 2
"""
function compact_filter(u, alpha; xmul::Int64=1, ymul::Int64=1)
    xlen = isqrt(div(size(u)[1]*xmul, ymul))
    ylen = div(size(u)[1], xlen)
    xI = sparse(Matrix(1.0I,xlen,xlen))
    yI = sparse(Matrix(1.0I,ylen,ylen))
    xLHS, xRHS = compact_filter_matrices(xlen, alpha)
    yLHS, yRHS = compact_filter_matrices(ylen, alpha)
    return kron(yLHS, xLHS) \ (kron(yRHS, xRHS) * u)
end

const _filter_memo = Dict()
function compact_filter_matrices(len::Int64, alpha, xmul::Int64=1, ymul::Int64=1)
    if (len,alpha) in keys(_filter_memo)
        return _filter_memo[(len, alpha)]
    end
    ext = [len-1:len; 1:len ; 1:2]
    a = 5.0/8.0 + 3.0*alpha/4.0
    b = alpha + 1.0/2.0
    c = alpha/4.0 - 1.0/8.0
    umap(i) = ext[i+2]

    LHS = Tuple{Int64, Int64, Float64}[] 
    RHS = Tuple{Int64, Int64, Float64}[] 
    for i in (1:len)
        row = umap(i)
        push!(LHS, (row, umap(i-1), alpha))
        push!(LHS, (row, umap(i), 1.0))
        push!(LHS, (row, umap(i+1), alpha))

        push!(RHS, (row, umap(i), a))
        push!(RHS, (row, umap(i+1), b/2.0))
        push!(RHS, (row, umap(i-1), b/2.0))
        push!(RHS, (row, umap(i+2), c/2.0))
        push!(RHS, (row, umap(i-2), c/2.0))
    end
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), len, len)
    RHS = sparse((x->x[1]).(RHS), (x->x[2]).(RHS), (x->x[3]).(RHS), len, len)
    _filter_memo[(len, alpha)] = (LHS,RHS)
    return LHS, RHS
end

function euler_rhs(r, ru, rv, rE, h)
    Frx, Fry, Frux, Fruy, Frvx, Frvy, FrEx, FrEy = euler_fluxes(r, ru, rv, rE)
    Fx = hcat(Frx, Frux, Frvx, FrEx)
    Fy = hcat(Fry, Fruy, Frvy, FrEy)
    return (-compact_div(Fx, Fy, h)[:,i] for i in (1:4))
end

function euler_rh4step(r, ru, rv, rE, h, k, alpha)
    f(u) = stack(euler_rhs(u[:,1], u[:,2], u[:,3], u[:,4], h))
    return [compact_filter(rk4(hcat(r,ru,rv,rE), f, k), alpha)[:,i] for i in (1:4)]
end

function rk4(u, f, k)
    du1 = f(u)
    du2 = f(u + (k/2.0)*du1)
    du3 = f(u + (k/2.0)*du2)
    du4 = f(u + k*du3)
    return u + (k/6.0)*(du1 + 2*du2 + 2*du3 + du4)
end

function euler_vortex(x, y, time, pars)
    γ  = 1.4
    rc = pars[1]
    ϵ  = pars[2]
    M₀ = pars[3]
    θ  = pars[4]
    x₀ = pars[5]
    y₀ = pars[6]

    r∞ = 1
    u∞ = 1
    E∞ = 1/(γ*M₀^2*(γ - 1)) + 1/2
    p∞ = (γ - 1) * (E∞ - 1/2)
    ubar = u∞ * cos(θ)
    vbar = u∞ * sin(θ)
    f = @. (1 - ((x - x₀) - ubar*time)^2 - ((y - y₀) - vbar*time)^2) / rc^2

    u = @. u∞ * (cos(θ) - ϵ*((y - y₀)-vbar*time) / (2π*rc) * exp(f/2))
    v = @. u∞ * (sin(θ) + ϵ*((x - x₀)-ubar*time) / (2π*rc) * exp(f/2))
    r = @. r∞ * (1 - ϵ^2 * (γ - 1) * M₀^2/(8π^2) * exp(f))^(1/(γ-1))
    p = @. p∞ * (1 - ϵ^2 * (γ - 1) * M₀^2/(8π^2) * exp(f))^(γ/(γ-1))

    ru = @. r*u
    rv = @. r*v
    rE = @. p/(γ - 1) + 1/2 * (ru^2 + rv^2) / r

    return [r, ru, rv, rE]
end

function vortex_test()
    pars = [0.5, 1, 0.5, pi/4, 2.5, 2.5]
    for alpha in [0.499, 0.48]
        errors = Float64[]
        hs = Float64[]
        nums = [32, 64, 128]
        for N in nums
            h = 10.0/N
            xs = [x*h for y in (0:N) for x in (0:N)] 
            ys = [y*h for y in (0:N) for x in (0:N)] 
            u0 = euler_vortex(xs, ys, 0, pars)
            T =ceil(Int64, 5*sqrt(2)/0.3/h)
            k = 5*sqrt(2)/T

            uref = [euler_vortex(xs, ys, t*k, pars) for t in (0:T)] 
            u = gen_euler(u0,h,k,alpha, T)
            push!(errors, maximum([abs(u[t][i][x] - uref[t][i][x]) 
                               for t in (1:T+1) for i in (1:4) for x in (1:(N+1)^2)])) 
            push!(hs, h)
            println(N)
        end
        # calculate linear regression
        clf()
        beta = [ones(length(nums)) log.(hs)] \ log.(errors)
        line = x->beta[1] + x*beta[2]
        xscale("log")
        yscale("log")
        scatter(hs, errors)
        plot(hs, exp.(line.(log.(hs))))
        xlabel("Grid Width h")
        ylabel("Estimated Linf Error u - u_ref")
        title("Grid Refinement Error for alpha=$(alpha)")
        regError = (log.(errors) - line.(log.(hs)))
        mean = sum(log.(errors))/length(nums)
        meanError = (x->x-mean).(log.(errors))
        R2 = 1 - (regError' * regError)/(meanError' * meanError)
        figtext(0.9, 0.15, "err = $(exp(beta[1])) * x^$(beta[2])", ha="right")
        figtext(0.9, 0.2, "R^2 = $(R2)", ha="right")
        
        show()
        savefig("error_plot_$(alpha).png")
     
    end
end

function gen_euler(u0, h, k, alpha, T)
    allu = [u0]
    for t in (1:T)
        r, ru, rv, rE = euler_rh4step(allu[t][1], allu[t][2], allu[t][3], allu[t][4], 
                                  h, k, alpha)
        push!(allu, [r,ru,rv,rE])
    end
    return allu
end

function mkanim(xs, ys, allu, N, index, skip, realtime, name)
    clf()
    animation = pyimport("matplotlib.animation")
    fig, ax = subplots(figsize=(6,4.5))
    function update(frame)
        ax.clear()
        ax.contourf(reshape(xs, N+1, N+1), reshape(ys,N+1,N+1), 
                    reshape(allu[skip*frame+1][index], N+1, N+1))
        ax.axis("equal")
        return (ax,)
    end
    ani = animation.FuncAnimation(fig, update, frames=floor(Int64, length(allu)/skip)-1, interval=50)
    writer = animation.PillowWriter(fps=ceil(length(allu)/skip/realtime), 
                                    bitrate=1800)
    ani.save(name, writer=writer)
    show()
end

function plotit(x, y, u, file, ti="")
    # Plotting
    clf()
    contourf(x, y, u, 0.8:0.03:1.01)
    axis("equal")

    title(ti)
    savefig(file)
    clf()
end

function test()
    pars = [0.5, 1, 0.5, pi/4, 2.5, 2.5]
    N = 64
    h = 10.0/N
    xs = [x*h for y in (0:N) for x in (0:N)] 
    ys = [y*h for y in (0:N) for x in (0:N)] 
    u0 = [i for i in euler_vortex(xs, ys, 0, pars)]
    println(typeof(u0))
    T =ceil(Int64, 10*sqrt(2)/0.3/h)
    k = 10*sqrt(2)/T
    uref = [euler_vortex(xs, ys, t*k, pars) for t in (0:T)] 
    return uref, gen_euler(uref[1],h,k,0.499, T), xs, ys, h, k, T
end

function kelvin_helmholtz()
    N = 256
    h = 1.0/N
    xs = [x*h for y in (0:N) for x in (0:N)] 
    ys = [y*h for y in (0:N) for x in (0:N)] 
    T =ceil(Int64, 1.0/0.3/h)
    k = 1.0/T
    r0 = [abs(y*h - 0.5) < 0.15 + sin(2*pi*x*h)/200.0 ? 2.0 : 1.0 for y in (0:N) for x in (0:N)]
    ru0 = [r*(r-1) for r in r0]
    rv0 = [0.0 for r in r0]
    rE0 = [3.0/(7.0/5.0 - 1.0) + 0.5*r*(r-1)^2 for r in r0]

    u = gen_euler([r0, ru0, rv0, rE0],h,k,0.48, T)
    return u, xs, ys, N

end

