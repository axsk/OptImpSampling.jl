using LinearAlgebra
using Zygote
using StochasticDiffEq, SciMLSensitivity
using StatsBase
using Lux, Random
using Optimisers
using BenchmarkTools

# dX = b + σ * v + σ * dB
# dY = -f - u'v + u'u/2
# Goal: find u = argmin Var[log(Y(T))] (which attains 0)
# Motivation: The optimal u-controlled process gives is a 0-variance estimator for
# W = ∫ f(X(t)) dt

# problem definition
b(X, t) = -X
v(X, t) = zero(X)
f(x, T) = 1.
sigma(X, t) = collect(I(length(X)))
ub = 1
lb = -1

function mydense(in=1)
    m = Lux.Chain(Lux.Dense(in,10,tanh), Lux.Dense(10,1))
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, m)
    return m, ps, st
end

function drift(dxy, xy, t, b, sigma, u, v, f)
    X = @view xy[1:end-1]
    let b = b(X, t),  # is this beautiful or horrible :D?
        v = v(X, t),
        u = u(X, t),
        σ = sigma(X, t),
        f = f(X, t)

        dxy[1:end-1] .= b .+ σ * v
        dxy[end] = - f - dot(u, v) + dot(u, u) / 2
    end
end

function noise(dxy, xy, t, sigma, u)
    dxy .= 0
    X = @view xy[1:end-1]
    dxy[1:end-1, 1:end-1] .= sigma(X, t)
    dxy[end, 1:end-1] .= u(X,t)
end

# stop after first component of trajectory crosses lower or upper bound
termination(ub, lb) = ContinuousCallback((u,t,int)->(u[1]-lb) * (ub-u[1]), terminate!)

function LogVarProblem(x0=[0.], T=10., luxmodel=mydense(); stoptime=true)
    model, ps, st = luxmodel
    xy0 = vcat(x0, 0.)
    n0 = zeros(length(xy0), length(xy0))

    p = Lux.ComponentArray(ps)
    u(p) = (X, t) -> model(X, p, st)[1]
    _drift(dxy, xy, p, t) = drift(dxy, xy, t, b, sigma, u(p), v, f)
    _noise(dxy, xy, p, t) = noise(dxy, xy, t, sigma, u(p))

    cb = stoptime ? termination(ub, lb) : nothing
    SDEProblem(_drift, _noise, xy0, T, p, noise_rate_prototype = n0, callback=cb)
end

function msolve(prob; ps=prob.p, dt=0.01, salg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), noisemixing=true))
    #prob = Zygote.@showgrad remake(prob, p=ps)  # this kills AD
    s = solve(prob, EM(), sensealg=salg, dt=dt, p=ps)
end

cost(sol) = sol[end][end]

function msens(prob)  # this works
    Zygote.gradient(ps->msolve(prob, ps=ps)|>cost, prob.p)[1]
end

function logvar(prob; ps=prob.p, n=10)  # this works
    #sum(_ -> msolve(prob, ps=ps) , 1:n)
    var(cost(msolve(prob, ps=ps)) for i in 1:n)
end

function dlogvar(prob; n=10)  # finally working
    Zygote.gradient(ps->logvar(prob, ps=ps, n=n), prob.p)[1]
end

function train(prob, learniter=10, mciter=10)
    params = prob.p
    rule = Optimisers.Adam()
    opt_state = Optimisers.setup(rule, params);  # optimiser state based on model parameters
    for i in 1:learniter
        lv, ∇params = withgradient(params) do p
            logvar(prob, ps=p, n=mciter)
        end
        ∇params = ∇params[1]  # why is ∇params = ([grad], )
        opt_state, params = Optimisers.update(opt_state, params, ∇params)
        prob = remake(prob, p=params)
        @show lv
    end

    return prob
end

function plotu(prob)
    model, ps, st = mydense()  # TODO: this needs to be given
    grid = -1:.1:1
    us = model(collect(grid)', prob.p, st)[1]
    plot(grid, us')
end

function benchmark()
    l = LogVarProblem()
    @show @benchmark msolve($l)
    @show @benchmark msens($l)
end

function test()
    l = LogVarProblem()
    msolve(l)
    msens(l)
    logvar(l)
    dlogvar(l)
    train(l)
    #plotu(l)
end

using Zygote
using StochasticDiffEq, SciMLSensitivity


# mwe for https://github.com/SciML/SciMLSensitivity.jl/issues/720
# errors in about 50% of cases
function mwe()
    x0 = [0.]
    drift(dx, x, p, t) = (dx .= p)
    noise(dx, x, p, t) = (dx .= 0.1)
    n0 = zeros(1,1)
    T = 100.
    p0 = [1.]
    #cb = ContinuousCallback((u,t,int)->(u[1]-1), terminate!)
    condition(u,t,integrator) = u[1] - 1
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition,affect!)
    prob = SDEProblem(drift, noise, x0, T, p0, noise_rate_prototype = n0, callback=cb)

    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), noisemixing=true)
    Zygote.gradient(p0) do ps
        solve(prob, EM(), dt=0.1, p=ps, sensealg=sensealg)[end][1]
    end
end

# mwe for https://github.com/FluxML/Zygote.jl/issues/1294
# doesnt reproduce atm
function mwe2()
    x0 = [0.]
    drift(dx, x, p, t) = (dx .= p)
    noise(dx, x, p, t) = (dx .= p)
    n0 = zeros(1,1)
    T = 1.
    p0 = [1.]
    prob = SDEProblem(drift, noise, x0, T, p0, noise_rate_prototype = n0)

    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), noisemixing=true)
    Zygote.gradient(p0) do ps
        solve(remake(prob, p=ps), EM(), dt=0.1, sensealg=sensealg)[end][1]
    end
end
