# optimal control for the isokann sampling problem
# we could start with fin elems / linear interp. of the eigenfunction
# and testing optimal sampling under the new criterion

# then progress to chi functions
# then replace fin elems by NN

# we need
# - a trajectory sampler + girsanov
# - the eigenfunction (test this before isokann convergence)
using Flux
using DifferentialEquations
using Interpolations
using LinearAlgebra

grad(f, x) = collect(Flux.gradient(f, x)[1])

doublewell(x) = ((x[1])^2 - 1) ^ 2

dynamics(;sigma=[1.], potential=doublewell, T=.1) = (;sigma, potential, T)

function sdeproblem(dynamics=dynamics(), x0=[0.])
    f(x,p,t) = - grad(dynamics.potential, x)  # find a way to use grads w/o [1]
    g(x,p,t) = dynamics.sigma
    prob = SDEProblem(f, g, x0, (0., dynamics.T))
end

function controlled_drift(xg, p, t)
    (;σ, k, U) = p
    x = @view xg[1:end-1]
    u = (-σ' * grad(k, x) / k(x)) :: Vector
    dV = (-Flux.gradient(U, x)[1]) :: Vector

    return [dV + u; sum(abs2, u) / 2] :: Vector
end

function controlled_noise(xg, p, t)
    (;σ, k, Σ, n) = p
    x = @view xg[1:end-1]
    Σ[n+1, 1:n] .= (-σ' * collect(Flux.gradient(k, x)[1]) / k(x)) :: Vector
    return Σ
end

function controlled_parameters(σ, k, U)
    n = size(σ,1)
    σ = reshape(σ, n, n)
    Σ = zeros(n+1, n+1)
    Σ[1:n, 1:n] = σ
    return (; n, σ, Σ, k, U)
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
    f(s[end][1]) * exp(-s[end][2]) - 1
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
    @show e.values[2]
    return LinearInterpolation(grid, v, extrapolation_bc=Flat())
end

function eigenfunction(grid=-2:.2:2, n=100, dynamics=dynamics())
    T = ulam(grid, n, dynamics, 1e-2)
    f = eigenfunction(grid, T)
    plot(grid, map(f, grid)) |> display
    T, f
end


### Statistics

using StatsBase
function expectation(f, x0, dynamics, n)
    prob = sdeproblem(dynamics, x0)

    ks = [f(solve(prob)[end]) for i in 1:n]
    k, v = mean_and_var(ks)
    return k, v
end
