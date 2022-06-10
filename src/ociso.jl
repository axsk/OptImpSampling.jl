# optimal control for the isokann sampling problem
# we could start with fin elems / linear interp. of the eigenfunction
# and testing optimal sampling under the new criterion

# then progress to chi functions
# then replace fin elems by NN

# we need
# - a trajectory sampler + girsanov
# - the eigenfunction (test this before isokann convergence)

# what we learned
# - need to adjust for time scaling of non-eigenfunction k
# - T eigenfunctions are hard, use SQRA

using Flux
using DifferentialEquations
using Interpolations
using LinearAlgebra
using Plots
using ForwardDiff
using Parameters

#grad(f, x) = collect(Flux.gradient(f, x)[1])

grad(f, x) = ForwardDiff.gradient(f, x)

doublewell(x) = ((x[1])^2 - 1) ^ 2



struct ProblemOptChi{P, C}
    potential::P
    T::Float64 # lag-time
    σ::Matrix{Float64} # noise
    Σ::Matrix{Float64} # augmented noise
    chi::C # chi function
    q::Float64 # rate of eigenfunction
    sl::Float64 # scale * lowerbound(shift)
end

ProblemOptChi(chi, q, sl) = ProblemOptChi(doublewell, 1., ones(1,1), ones(2,2), chi, q, sl)

function ProblemOptChi()
    T, f, v, val = eigenfunction()
    sl = - minimum(v)
    chi(x) = f(x[1]) + sl
    q = -log(real(val))
    ProblemOptChi(chi, q, sl)
end

function u_star(x, t, T, σ, chi, q, sl)
    λ = exp(-q * (T-t))
    u = σ' * grad(chi, x) / (chi(x) + sl / λ)
    return u
end

u_star(x, p::ProblemOptChi, t) = u_star(x, t, p.T, p.σ, p.chi, p.q, p.sl)

function controlled_drift(xg, p::ProblemOptChi, t)
    x = @view xg[1:end-1]
    u = u_star(x, p, t)
    du = [-grad(p.potential, x) + u; sum(abs2, u) / 2]
    return du
end

function controlled_noise(xg, p::ProblemOptChi, t)
    x = @view xg[1:end-1]
    n = length(x)
    p.Σ[n+1, 1:n] .= u_star(x, p, t)
    return p.Σ :: Matrix
end

function SDEProblem(p::ProblemOptChi, x0)
    SDEProblem(controlled_drift, controlled_noise, [x0; 0.], (0., p.T), p, noise_rate_prototype = p.Σ)
end

function evaluate(p::ProblemOptChi, x0)
    sde = SDEProblem(p, x0)
    s = solve(sde)
    e = p.chi(s[end][1]) * exp(-s[end][2]) - p.sl
    return e
end


### COMPUTATION OF eigenfunction

function gridcell(grid, x)
    for i in 1:length(grid)-1
        if (grid[i] + grid[i+1]) / 2 > x
            return i
        end
    end
    return length(grid)
end

@assert gridcell(1:3, 1.6) == 2
@assert gridcell(1:3, 0.3) == 1
@assert gridcell(1:3, 7) == 3

function ulam(grid, n, dynamics, tol=1e-3)
    N = length(grid)
    T = zeros(N,N)
    for (i, x) in enumerate(grid)
        prob = sdeproblem(dynamics, [x])
        for nn in 1:n
            y = solve(prob, abstol=tol, reltol=tol)[end]
            j = gridcell(grid, y[1])
            T[i, j] += 1
        end
    end
    T = T ./ sum(T, dims=2)
end

function eigenfunction(grid, T::Matrix)
    e = eigen(T, sortby=x->-real(x))
    v = e.vectors[:, 2] |> real
    @show val = e.values[2]
    int = CubicSplineInterpolation(grid, v, extrapolation_bc=Flat())
    return int, v, val
end

function eigenfunction(grid=-2:.2:2, n=100, dynamics=dynamics())
    T = ulam(grid, n, dynamics, 1e-2)
    f, v, val = eigenfunction(grid, T)
    plot(grid, map(f, grid)) |> display
    T, f, v, val
end

#= legacy code

### Statistics

dynamics(;sigma=[1.], potential=doublewell, T=.1) = (;sigma, potential, T)

function sdeproblem(dynamics=dynamics(), x0=[0.])
    f(x,p,t) = - grad(dynamics.potential, x)
    g(x,p,t) = dynamics.sigma
    prob = SDEProblem(f, g, x0, (0., dynamics.T))
end

u_star(x, k, σ) = (σ' * grad(k, x) / k(x)) :: Vector
u_star(x, k, σ) = (σ' * grad(k, x) / k(x)) :: Vector

function controlled_drift(xg, p, t)
    (;σ, k, U, u) = p
    x = @view xg[1:end-1]
    uu = u(x)
    du = [-grad(U, x) + uu; sum(abs2, uu) / 2]

    return du
end

function controlled_noise(xg, p, t)
    (;σ, k, Σ, n, u) = p
    x = @view xg[1:end-1]
    Σ[n+1, 1:n] .= u(x)
    return Σ :: Matrix
end

function controlled_parameters(σ, k, U)
    n = size(σ,1)
    σ = reshape(σ, n, n)
    Σ = zeros(n+1, n+1)
    Σ[1:n, 1:n] = σ
    u = x -> u_star(x, k, σ)
    return (; n, σ, Σ, k, U, u)
end

using StatsBase
function expectation(f, x0, dynamics, n)
    prob = sdeproblem(dynamics, x0)

    ks = [f(solve(prob)[end]) for i in 1:n]
    k, v = mean_and_var(ks)
    return k, v
end


controlled_problem(; σ=[1], U=doublewell, k=k1, x0 = [0.], T = 1) = controlled_problem(σ, U, k, x0, T)

function controlled_problem(σ, U, k, x0, T)
    p = controlled_parameters(σ, k, U)
    SDEProblem(controlled_drift, controlled_noise, [x0; 0], (0., T), p, noise_rate_prototype = ones(p.n+1, p.n+1))
end

# leads to zero control
k1(x) = 0 * sum(x) + 1 # positive for division, 0 mult for zygote to not return nothing

function optexp(f)
    k(x) = f(x[1]) + 1 # shifted eigenfunction
    p = controlled_problem(k=k)
    s = solve(p)
    plot(s) |> display

    k(s[end][1]) * exp(-s[end][2]) - 1 # this should be 0 variance for K[f]
    k, p, s
end

function compare(nef = 1000, nsol = 10000)
    T, f = eigenfunction(-2:.1:2, 2000) # compute the eigenfunction
    k(x) =  f(x[1]) + 1 # shifted eigenfunction
    #k(x) = f(x[1]) - minimum(f.(-3:3)) + 0.1
    p0 = controlled_problem(ones(1,1), doublewell, x->1, 0., 1.)
    pu = controlled_problem(ones(1,1), doublewell, k, 0., 1.)
    sol0=[solve(p0) for i in 1:10000]
    solu=[solve(pu) for i in 1:10000]

    map([solu, sol0]) do sol
        ex = eval_girsanov.(sol, k)
       @show mean_and_std(ex)
       histogram(ex)|>display
    end
    sol0, solu, f
end




eval_girsanov(s::DifferentialEquations.RODESolution, k) = eval_girsanov(s[end][1], s[end][2], k)

eval_girsanov(x, g, k) = k(x) * exp(-g)

=#
