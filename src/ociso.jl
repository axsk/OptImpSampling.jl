# optimal control for the isokann sampling problem
# we start with fin elems / linear interp. of the eigenfunction
# and testing optimal sampling under the new criterion


#using Flux
using DifferentialEquations
#using StochasticDiffEq
using Interpolations
using LinearAlgebra
using Plots
using ForwardDiff
using Parameters

import Plots: plot
import DifferentialEquations.solve
import StatsBase.mean_and_std
using Parameters
using Zygote
using StaticArrays

#grad(f, x) = collect(Flux.gradient(f, x)[1])
fgrad(f, x) = ForwardDiff.gradient(f, x)
#grad(f,x) = Zygote.gradient(f, x)[1]

global cache = Dict{Tuple{DataType, Int64}, ForwardDiff.GradientConfig}()
global temp = Dict{Tuple{DataType, Int64}, Vector{Float64}}()

function cfgrad(f, x)
    df = get!(temp, (typeof(f), length(x))) do
        copy(x)
    end :: Vector{Float64}
    cfg = get!(cache, (typeof(f), length(x))) do
        ForwardDiff.GradientConfig(f, x)
    end :: ForwardDiff.GradientConfig
    ForwardDiff.gradient!(df, f, x, cfg)
    df::Vector{Float64}
end

grad(f, x) = fgrad(f, x)

abstract type Diff end
struct Differentiator2{F,T,R} <: Diff
    f::F
    cfg::T
    res::R
end
Diff(f, x::AbstractArray) = Differentiator2(f, ForwardDiff.GradientConfig(f, x), copy(x))

(d::Diff)(x) = d.f(x)
function grad(d::Diff, x)
    ForwardDiff.gradient!(d.res, d.f, x, d.cfg)
    d.res
end
grad!(df, d::Diff, x) = ForwardDiff.gradient!(df, d.f, x, d.cfg)


doublewell(x) = ((x[1])^2 - 1) ^ 2

abstract type ProblemOptControl end
@with_kw mutable struct ProblemOptChi4{P, C} <: ProblemOptControl
    potential::P            = doublewell
    T::Float64              = 1              # lag-time
    σ::Matrix{Float64}      = ones(1,1)      # noise
    Σ::Matrix{Float64}      = zeros(2,2)     # augmented noise
    chi::C                                   # chi function
    q::Float64                               # rate of eigenfunction
    b::Float64                               # scale * lowerbound(shift)
    dt::Float64             = 0.01
    forcing::Float64        = 1.
end

@with_kw mutable struct ProblemOptCached{P, C, CA} <: ProblemOptControl
    potential::P            = doublewell
    T::Float64              = 1              # lag-time
    σ::Matrix{Float64}      = ones(1,1)      # noise
    Σ::Matrix{Float64}      = zeros(2,2)     # augmented noise
    chi::C                                   # chi function
    q::Float64                               # rate of eigenfunction
    b::Float64                               # scale * lowerbound(shift)
    dt::Float64             = 0.01
    forcing::Float64        = 1.
    derivatives::CA
end

derivatives(U, chi, x) = (; dU = Diff(U, x), dchi = Diff(chi, x))

ProblemOptChi(chi, q, b) = ProblemOptChi4(doublewell, 1., ones(1,1), collect(Diagonal([1.,0])), chi, q, b, 0.01, 1.)
ProblemOptChi(chi, q, b) = ProblemOptCached(doublewell, 1., ones(1,1), collect(Diagonal([1.,0])), chi, q, b, 0.01, 1., derivatives(doublewell, chi, ones(1)))

function plot_grad_and_u(p, grid=-2:.05:2)
    plot(grid, x->grad(p.chi, [x])[1], label="grad")
    plot!(grid, x->control(p, [x],0)[1], label ="u*")
end

function ProblemOptChi(;step=0.1, grid=-2:step:2)
    f, v, q = eigenfunction_sqra(grid=grid)
    b = -minimum(v) + .1
    chi(x) = f(x[1]) + b
    ProblemOptChi(chi, q, b)
end

global linforce :: Float64 = 1.

control(p::ProblemOptControl, x::AbstractVector, t) = control(x, t, p.T, p.σ, p.chi, p.q, p.b, p.derivatives.dchi)

function control(x, t, T, σ, chi, q, b, dchi)
    @assert q<0
    λ = exp(q * (T-t))
    #x = SVector{1}(x)
    u = grad(dchi, x)
    #u = grad(dchi, x)
    u .= σ' * u
    u .*=  linforce / (chi(x) - b + b / λ)
    return u
end

function controlled_drift(du, xg::Vector{Float64}, p::ProblemOptControl, t)
    x = @view xg[1:end-1]
    u = control(p, x, t)
    #du[1:end-1] = - grad(p.potential, SVector{1}(x))
    du[1:end-1] = - grad(p.derivatives.dU, x)
    #@view(du[1:end-1]) .+= SMatrix{1,1}(p.σ) * u  # eq. (5)
    @view(du[1:end-1]) .+= p.σ * u  # eq. (5)
    du[end] = sum(abs2, u) / 2  # eq. (19)
end

function controlled_noise(dg, xg, p::ProblemOptControl, t)
    x = @view xg[1:end-1]
    dg .= 0  # maybe unnecessary
    dg[1:end-1, 1:end-1] .= p.σ  # eq. (5)
    dg[end, 1:end-1] .= control(p, x, t)  # eq. (19)
end

function SDEProblem(p::ProblemOptControl, x0)
    StochasticDiffEq.SDEProblem(controlled_drift, controlled_noise,
        [x0; 0.], (0., p.T), p, noise_rate_prototype = p.Σ)
end

solve(p::ProblemOptControl, x0; showplot=false, solver=SROCK2(), dt=p.dt) = solve(SDEProblem(p, x0), solver, dt=dt)

## Utility functions

function plot(p::ProblemOptControl, sol)
    plot(sol, label = ["X_t" "G"])
    plot!(sol.t, control(p, sol), label = "u") |> display
end

function evaluate(p::ProblemOptControl, x0)
    s = solve(p, x0)
    e = p.chi(s[end][1]) * exp(-s[end][2]) - p.b
    return e
end

function control(p::ProblemOptControl, sol::SciMLBase.AbstractODESolution)
    us = []
    for (t,u) in zip(sol.t, sol.u)
        u = control(p, u[1:end-1], t)
        push!(us,u[1])
    end
    return us
end

mean_and_std(p::ProblemOptControl, x0, n) = mean_and_std([evaluate(p, x0) for i in 1:n])

function test()
    p = ProblemOptChiSqra()
    mean_and_std(p, 0., 100)
end

## Eigenfunction via SQRA

using Sqra

function eigenfunction_sqra(; grid=-2:.2:2, potential=doublewell, sigma=1)
    beta = 2 / sigma^2  # Einstein relation
    u = map(potential, grid)
    u = reshape(u, length(grid), 1)
    Q = Sqra.sqra(u, beta) * (1/step(grid))^2 / beta
    e = eigen(collect(Q), sortby=x->-real(x))
    vec = e.vectors[:,2]
    val = e.values[2]
    int = CubicSplineInterpolation(grid, vec, extrapolation_bc=Flat())
    return int, vec, val
end

### DEPRECATED: computation of eigenfunction by ulam (sqra is faster)

function ProblemOptChiUlam(; n=300)
    f, v, q= eigenfunction(n=n)
    b = -minimum(v) + .1
    chi(x) = f(x[1]) + b
    ProblemOptChi(chi, q, b)
end

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

function eval_std(p, x0)
    s=sdeproblem(dynamics(), [x0])
    sol = solve(s)
    p.chi(sol[end]) - p.b
end

function eigenfunction(grid, T::Matrix)
    e = eigen(T, sortby=x->-real(x))
    v = e.vectors[:, 2] |> real
    @show val = e.values[2]
    int = CubicSplineInterpolation(grid, v, extrapolation_bc=Flat())
    plot(grid, x->int(x)) |> display
    return int, v, val
end

function eigenfunction(; grid=-2:.2:2, n=300, dynamics=dynamics())
    T = ulam(grid, n, dynamics, 1e-2)
    int, vec, val = eigenfunction(grid, T)
    q = log(real(val)) / dynamics.T
    return int, vec, q
end

dynamics(;sigma=[1.], potential=doublewell, T=.1) = (;sigma, potential, T)

function sdeproblem(dynamics=dynamics(), x0=[0.])
    f(x,p,t) = - grad(dynamics.potential, x)
    g(x,p,t) = dynamics.sigma
    prob = StochasticDiffEq.SDEProblem(f, g, x0, (0., dynamics.T))
end
