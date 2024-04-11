# I guess I should be putting my name on this: Ryan Martinez    
"""
    Inputs two functions R0 and R1 of 1 variable and interpolates them 
    to a function of 2 variables where i is the variable of interpolation
"""
function int(R0, R1, i)
    j = 3 - i
    R(xy) = (1.0 - xy[i])* R0(xy[j]) + xy[i]*R1(xy[j])
    return R
end

"""
    The hermite interpolation between R0 (which now is a matrix containing the tangent 
    vectors as well
"""
function hermite(R0, R1, i)
    j= 3 - i
    R(xy) = [((2.0*xy[i]^3 - 3.0*xy[i]^2 + 1.0)*R0(xy[j])[:,1]
             +(xy[i]^3 - 2.0*xy[i]^2 + xy[i])*R0(xy[j])[:,2]
             +(-2.0*xy[i]^3 + 3.0*xy[i]^2)*R1(xy[j])[:,1]
             +(xy[i]^3 - xy[i]^2)*R1(xy[j])[:,2]) ((6.0*xy[i]^2 - 6.0*xy[i])*R0(xy[j])[:,1]
             +(3.0*xy[i]^2 - 4.0*xy[i] + 1.0)*R0(xy[j])[:,2]
             +(-6.0*xy[i]^2 + 6.0*xy[i])*R1(xy[j])[:,1]
             +(3.0*xy[i]^2 - 2.0*xy[i])*R1(xy[j])[:,2])]
    return R
end

"""
    Inputs 4 boundary functions in the order Up, Down, Left, Right 
    which are parametrized from 0 to 1 in the direction 
    of increasing x or y. In particular these functions MUST 
    agree at the corners to get what is expected Fu(0) = Fl(1), 
    Fu(1) = Fr(1), Fd(0) = Fl(0), Fd(1) = Fr(0); otherwise we take the average. 
    
    Returns a 2D interpolation using transfinite interpolation 
    (couldn't quite get the hermite to work this way, but I left in the generality.
    I think the issue is that the normals don't agree at the corners and that causes 
    Hermite to do a quirky little bend thing)
"""
function tfi(Fd, Fu, Fl, Fr, func)
    P = func(x->func(Fl, Fr, 1)([x,0.0]), x->func(Fl, Fr,1)([x,1.0]), 2)
    Q = func(y->func(Fd, Fu, 2)([0.0,y]), y->func(Fd, Fu,2)([1.0,y]), 1)
    return xy -> func(Fl, Fr, 1)(xy)  + func(Fd, Fu, 2)(xy) - 0.5*(P(xy) + Q(xy))
end


"""
    This is the version of the above function that will work for the hermite guy
"""
function tfi_h(Fd, Fu, Fl, Fr, func)
    return xy -> func(Fd, Fu, 2)(xy)
end

"""
    Simply applying the tfi function above to the given example 
"""
function tfi_linear(xy, A=0.4)
    return tfi(x->[x,64.0*A*x^3*(1.0-x)^3], 
               x->[x,1+A*x^3*(6.0*x^2 - 15.0*x +10.0)], 
               y->[0.0,y], y->[1.0,(1.0 + A)*y], int)(xy)
end

"""
    Simply applying the tfi_h function above to the given example 
"""
function tfi_orthogonal(xy, A=0.4, T=0.5)
    return tfi_h(x->[[x,64.0*A*x^3*(1.0-x)^3] T*unit([6.0*64.0*A*(x-0.5)*x^2*(x-1.0)^2, 1.0])], 
               x->[[x,1+A*x^3*(6.0*x^2 - 15.0*x +10.0)] T*unit([-30.0*A*x^2*(x-1.0)^2, 1.0])], 
               y->[[0.0,y] T*[1.0, 0.0]], y->[[1.0,(1.0 + A)*y] T*[1.0, 0.0]], hermite)(xy)[:,1]
end

"""
    The unit vector in the direction of v
"""
function unit(v)
    return 1.0/sqrt((v')*v)*v
end


"""
    Takes a matrix (although I feel like a vector of vectors is more natural but whatever)
    pv whose rows are coordinates of boundary points in order around the boundary 
    and triangulates with roughly Delaunay triangles of max area less than hmax^2/2^(nref+1) 
    and side length approximately hmax/2^(nref)
"""
function pmesh(pv, hmax::Float64, nref)
    ps = Vector{Float64}[] # this will be a vector of points 
    for i in (1:size(pv)[1]-1)
        dist = (x->sqrt(x'*x))(pv[i,:] - pv[i+1,:]) # the distance between point i and i+1
        n = ceil(dist / hmax)
        append!(ps, [pv[i,:]*(1.0 - k/n) + pv[i+1,:]*k/n for k in (0:n-1)]) #linear interpolation
    end
    while true
        t = map(x -> collect(x), D.triangulate(ps).triangles) # collect the tuples into vectors
        t = filter(x -> inpolygon(masscenter(ps[x]), pv) && area(ps[x]) >= 10^(-12), t)
                   
        largest = findfirst(x -> area(ps[x]) > hmax^2/2, t)
        if isnothing(largest)
            for i in (1:nref)
                edges, _, _ = all_edges(hcat(t...)')
                for e in [edges[i, :] for i in (1:size(edges)[1])]
                    push!(ps, masscenter(ps[e]))
                end
                t = map(x -> collect(x), D.triangulate(ps).triangles)
                t = filter(x -> inpolygon(masscenter(ps[x]), pv) && area(ps[x]) >= 10^(-12), t)
            end
            return hcat(ps...)', hcat(t...)', boundary_nodes(hcat(t...)')
        end
        push!(ps, circumcenter(ps[t[largest]]))
    end
end

"""
    The circumcenter of a triangle to be sure Delaunay deletes the bad triangle. The circumcenter 
    is the unique point that makes lies on the perpendicular bisector of all lines, which is 
    a quick matrix inversion

    I'm not sure we're supposed to be using the circumcenter right? What 
    if Delaunay makes a big triangle on the border, then the circumcenter 
    is (maybe) outside the convex hull of the shape?!
"""
function circumcenter(ps)
    m12 = (ps[1] + ps[2])/2.0
    m13 = (ps[1] + ps[3])/2.0
    d12 = (ps[1] - ps[2])
    d13 = (ps[1] - ps[3])
    return hcat(d12, d13)' \ [d12' * m12, d13' * m13]
end

"""
    To check outside triangles we use the center of mass which is always in the interior of a 
    triangle, although I'm realizing now that this might not work in general. I guess it's fine 
    because Delaunay will always make a planar graph so any triangle is either completely 
    inside or outside the desired shape (but only if Delaunay keeps the edges?) Whatever, 
    time to turn in
"""
function masscenter(ps)
    return sum(ps)/length(ps)
end

"""
    Use the wedge product to compute the area of a triangle
"""
function area(ps)
    d12 = (ps[1] - ps[2])
    d13 = (ps[1] - ps[3])
    return 0.5 * abs(d12[1]*d13[2] - d12[2]*d13[1])
end


### Below is the given mesh utilities

# Various mesh utilities
# UC Berkeley Math 228B, Per-Olof Persson <persson@berkeley.edu>

using PyPlot
import Delaunator as D

"""
    plot_mapped_grid(R, n1, n2=n1)

Create a Cartesian grid of dimensions (n1+1)-by-(n2+1) on the unit square.
Map it by the functions xy = R(ξη). Plot the resulting structured grid.

Example:

    identity_map(ξη) = ξη
    plot_mapped_grid(identity_map, 40);
"""
function plot_mapped_grid(R, n1, n2=n1, file="fig.png")
    clf()
    xy = Vector{Float64}[ R([ξ,η]) for ξ in (0:n1)./n1, η in (0:n2)./n2 ]
    x,y = first.(xy), last.(xy)
    axis("equal")
    plot(x, y, "k", x', y', "k", linewidth=1)
    savefig(file)
    clf()
end

"""
    t = delaunay(p)

Delaunay triangulation `t` of N x 2 node array `p`.
"""
delaunay(p) = collect(reinterpret(reshape, Int32, 
        D.triangulate(D.PointsFromMatrix(p')).triangles)')

"""
    edges, boundary_indices, emap = all_edges(t)

Find all unique edges in the triangulation `t` (ne x 2 array)
Second output is indices to the boundary edges.
Third output emap (nt x 3 array) is a mapping from local triangle edges
to the global edge list, i.e., emap[it,k] is the global edge number
for local edge k (1,2,3) in triangle it.
"""
function all_edges(t)
    etag = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t,1))
    etag = sortslices(etag, dims=1)
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = etag[keep,1:2]
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end

"""
    e = boundary_nodes(t)

Find all boundary nodes in the triangulation `t`.
"""
function boundary_nodes(t)
    edges, boundary_indices, _ = all_edges(t)
    return unique(edges[boundary_indices,:][:])
end

"""
    tplot(p, t, u=nothing)

If `u` == nothing: Plot triangular mesh with nodes `p` and triangles `t`.
If `u` == solution vector: Plot filled contour color plot of solution `u`.
"""
function tplot(p, t, u=nothing, file="fig.png")
    clf()
    axis("equal")
    if u == nothing
        tripcolor(p[:,1], p[:,2], Array(t .- 1), 0*t[:,1],
                  cmap="Set3", edgecolors="k", linewidth=1)
    else
        tricontourf(p[:,1], p[:,2], Array(t .- 1), u[:], 20)
    end
    draw()
    savefig(file)
    clf()
end

"""
    inside = inpolygon(p, pv)

Determine if each point in the N x 2 node array `p` is inside the polygon
described by the NE x 2 node array `pv`.
"""
function inpolygon(p, pv)
    if ndims(p) == 2 && size(p,2) == 2
        return [ inpolygon(p[ip,:], pv) for ip = 1:size(p,1) ]
    end
    cn = 0
    for i = 1:size(pv,1) - 1
        if pv[i,2] <= p[2] && pv[i+1,2] > p[2] ||
           pv[i,2] > p[2] && pv[i+1,2] <= p[2]
            vt = (p[2] - pv[i,2]) / (pv[i+1,2] - pv[i,2])
            if p[1] < pv[i,1] + vt * (pv[i+1,1] - pv[i,1])
                cn += 1
            end
        end
    end
    return cn % 2 == 1
end
