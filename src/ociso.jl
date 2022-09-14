# optimal control for the isokann sampling problem
# we start with fin elems / linear interp. of the eigenfunction
# and testing optimal sampling under the new criterion

export ProblemOptChi, doublewell, triplewell, msolve, evaluate, mean_and_std

#using Flux
using DifferentialEquations
#using StochasticDiffEq
using Interpolations
using LinearAlgebra
using Plots
using ForwardDiff
using Parameters

import Plots: plot
#import DifferentialEquations.solve
import StatsBase.mean_and_std
using Parameters
using Zygote
using StaticArrays
using Arpack

xgrad(f, x) = collect(Flux.gradient(f, x)[1])
fgrad(f, x) = ForwardDiff.gradient(f, x)
zgrad(f,x) = Zygote.gradient(f, x)[1]

grad(f, x) = fgrad(f, x)

doublewell(x) = ((x[1])^2 - 1) ^ 2

function triplewell(u)
    x, y = u
    V =  (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end

dim(::typeof(doublewell)) = 1
dim(::typeof(triplewell)) = 2

abstract type ProblemOptControl end
@with_kw mutable struct ProblemOptChi11{N, P, C, NN} <: ProblemOptControl
    potential::P             = doublewell
    T::Float64               = 1              # lag-time
    σ::SMatrix{N,N, Float64, NN} = SMatrix{dim(potential),dim(potential),Float64}(I)      # noise
    chi::C                                   # chi function
    q::Float64                               # rate of eigenfunction
    b::Float64                               # scale * lowerbound(shift)
    dt::Float64              = 0.01
    forcing::Float64         = 1.
end

ProblemOptChi = ProblemOptChi11

Sigma(sigma) = similar(sigma, size(sigma).+1 ...)

derivatives(U, chi, x) = (; dU = Diff(U, x), dchi = Diff(chi, x))

#ProblemOptChi(chi, q, b) = ProblemOptChi4(doublewell, 1., ones(1,1), collect(Diagonal([1.,0])), chi, q, b, 0.01, 1.)

function plot_grad_and_u(p, grid=-2:.05:2)
    plot(grid, x->grad(p.chi, [x])[1], label="grad")
    plot!(grid, x->control(p, [x],0)[1], label ="u*")
end

function ProblemOptSqra(;step=0.1, grid=-2:step:2, kwargs...)
    f, v, q = eigenfunction_sqra(grid=grid)
    b = -minimum(v) + .1
    chi(x) = f(x[1]) + b
    ProblemOptChi(chi=chi, q=q, b=b; kwargs...)
end

function control(p::ProblemOptChi{N}, x::AbstractVector, t) where {N}
    control(x, t, p.T, p.σ, p.chi, p.q, p.b, p.forcing, Val(N))
end

# optimal control assuming χ = ϕ + b with Kϕ = λϕ and λ = exp(tq)
function control(x, t, T, σ, χ, q, b, forcescale, _::Val{N}) where {N}
    forcescale == 0. && return zero(SVector{N})
    @assert q <= 0
    t>T && (t=T)
    @assert t<=T

    λ = exp(q * (T-t))
    logψ(x) = log(λ*(χ(x)-b) + b)
    if λ*(χ(x)-b) + b <= 0
        @show χ(x), λ, b
        @assert χ(x) > 0
    end
    u = forcescale * σ' * grad(logψ, x)
    return u
end

statify(p::ProblemOptChi{N}, x::AbstractVector) where {N} = SVector{N}(x)

function controlled_drift(du, xg::Vector{Float64}, p::ProblemOptControl, t)
    x = statify(p, @view xg[1:end-1])
    u = control(p, x, t)
    du[1:end-1] = - grad(p.potential, x)
    @view(du[1:end-1]) .+= p.σ * u  # eq. (5)
    du[end] = sum(abs2, u) / 2  # eq. (19)
end

function controlled_noise(dg, xg, p::ProblemOptControl, t)
    x = statify(p, @view xg[1:end-1])
    dg[1:end-1, 1:end-1] .= p.σ  # eq. (5)
    dg[end, 1:end-1] .= control(p, x, t)  # eq. (19)
    dg[:, end] .= 0
end

function SDEProblem(p::ProblemOptControl, x0)
    StochasticDiffEq.SDEProblem(controlled_drift, controlled_noise,
        [x0; 0.], (0., p.T), p, noise_rate_prototype = Sigma(p.σ))
end

function msolve(p::ProblemOptControl, x0; solver=SROCK2(), dt=p.dt)
    prob = SDEProblem(p, x0)
    solve(prob, solver, dt=dt)
end

function msolve(p::ProblemOptControl, x0, n, solver=SROCK2())
    prob = SDEProblem(p, x0)  # custom SDEProblem constructor
    #=
    @time int = init(prob, solver, dt=p.dt)
    map(1:n) do i
        @time reinit!(int)
        @time xg = solve!(int)[end] :: Vector{Float64}
        xg[end] = exp(-xg[end])
    end
    =#
    [solve(prob, solver, dt=p.dt)[end] for i in 1:n]  # TODO: msolve above does not return end time
end



function evaluate(p::ProblemOptControl, x0::AbstractVector, n)
    return map(msolve(p, x0, n)) do s
        x = @view s[1:end-1]
        @assert all(isfinite.(x))
        y = p.chi(x)
        @assert isfinite(y)
        e = y * exp(-s[end])  # - p.b
        @assert isfinite(e)
        return e
    end
end

evaluate(p::ProblemOptControl, x0::AbstractVector) = evaluate(p, x0, 1)[1]

function prop_and_evaluate(p::ProblemOptControl, x0::AbstractVector, n)
    d = length(x0)
    ys = similar(x0, d, n)
    ws = zeros(n)
    for i in 1:n
        s = msolve(p, x0)[end]
        ys[:, i] .= s[1:end-1]
        ws[i] = exp(-s[end])
    end
    chis = mapslices(p.chi, ys, dims=1)
    return ys, chis, ws
end


## Plots

function plotconvbig()
    plotconvergence(n=1000, steps=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, .5], dts=[.1, 0.01, 0.001, 0.0001, 0.00001])
    savefig("plotconvbig.png")
end
plotconvsmall() = plotconvergence(n=10, steps=[0.01, 0.1, 0.2], dts=[.1, .01])

function plotconvergence(;n=100, steps=[.01, .03, .1, .3], dts=[.001, .003, .01, .03, .1], kwargs...)
    p = plot(xlabel="dx", ylabel="std", xaxis=:log, yaxis=:log, legend=:bottomright)

    @time for dt in dts
        ts = []
        stds = map(steps) do step

            t = @elapsed mean, std = mean_and_std(ProblemOptSqra(;step=step, dt=dt, kwargs...),[0.], n)
            @show dt, step, t
            push!(ts, t)
            std
        end
        plot!(p, steps, stds, label="dt=$dt", markersize=sqrt.(ts), markershape=:circle) |> display
    end

    p
end

## Utility functions

function plot(p::ProblemOptControl, sol)
    plot(sol, label = ["X_t" "G"])
    plot!(sol.t, control(p, sol), label = "u") |> display
end


function control(p::ProblemOptControl, sol::SciMLBase.AbstractODESolution)
    us = []
    for (t,u) in zip(sol.t, sol.u)
        u = control(p, u[1:end-1], t)
        push!(us,u[1])
    end
    return us
end

mean_and_std(p::ProblemOptControl, x0, n) = mean_and_std(evaluate(p, x0, n))

## Eigenfunction via SQRA

function eigenfunction_sqra(; grid=-2:.2:2, potential=doublewell, sigma=1)
    beta = 2 / sigma^2  # Einstein relation
    u = map(potential, grid)
    u = reshape(u, length(grid), 1)
    Q = sqra(u, beta) * (1/step(grid))^2 / beta
    val, vec = eigs(Q, which=:SM, nev=2)
    vec = vec[:,2] |> real
    val = val[2] |> real
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
