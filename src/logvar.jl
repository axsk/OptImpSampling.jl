using LinearAlgebra
using Zygote
using StochasticDiffEq, SciMLSensitivity
using StatsBase: var
using Random
import Lux
import Optimisers
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

function mydense(dim=1)
    m = Lux.Chain(Lux.Dense(dim,10,tanh), Lux.Dense(10,dim))
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

function LogVarProblem(x0=[0.], T=10., luxmodel=mydense(); stoptime=false)
    model, ps, st = luxmodel
    xy0 = vcat(x0, 0.)
    n0 = zeros(length(xy0), length(xy0))

    p = Lux.ComponentArray(ps)
    u(p) = (X, t) -> model(X, p, st)[1]
    _drift(dxy, xy, p, t) = drift(dxy, xy, t, b, sigma, u(p), v, f)
    _noise(dxy, xy, p, t) = noise(dxy, xy, t, sigma, u(p))

    cb = stoptime ? termination(ub, lb) : nothing
    StochasticDiffEq.SDEProblem(_drift, _noise, xy0, T, p, noise_rate_prototype = n0, callback=cb)
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

function test_logvar()
    l = LogVarProblem()
    msolve(l)
    msens(l)
    logvar(l)
    dlogvar(l)
    train(l)
    #plotu(l)
end

### second attempt
# construct X and Y seperately
# allows reusing forward trajs


function primal_process(x0=[0.], T=1., ub=1., lb=1.)
    drift(x, p, t) = let b = b(x, t), v = v(x, t), σ = sigma(x, t)
        b .+ σ * v
    end

    noise(x, p, t) = sigma(x, t)[1]

    cb = termination(ub, lb)
    StochasticDiffEq.SDEProblem(drift, noise, x0, T, callback = cb, save_noise = true,
        alg=EM(), dt=.01)
end

function adjoint_process(X, luxmodel)
    model, ps, st = luxmodel
    p = Lux.ComponentArray(ps)
    u(x, p, t) = model(x, p, st)[1]  # TODO: augment time to state

    drift(x, p, t) = let X = X(t),
        u = u(X, p, t),
        v = v(X, t),
        f = f(X, t)
        - dot(u,v) - f + 1/2 * dot(u, u)
    end

    noise(x, p, t) = - u(X(t), p, t)'

    noisecopy = NoiseWrapper(X.W)  # this is not differentiable
    noisecopy = deepcopy(X.W)  # i cant remember whether this worked
    StochasticDiffEq.SDEProblem(drift, noise, 0., X.prob.tspan[end], p,
        noise=noisecopy, noise_rate_prototype=zeros(1,2), alg=EM(), dt=.01)
end

function solveXY(x0=[0.], luxmodel=mydense(length(x0)), T=1., ub=1., lb=1., dt=.1)
    X = solve(primal_process(x0, T, ub, lb), EM(), dt=dt)
    Y = solve(adjoint_process(X, luxmodel), EM(), dt=dt)
end

function Yensemble(;n=10, x0=[0.], luxmodel=mydense(length(x0)), T=1., ub=1., lb=1., dt=.1)
    map(1:n) do _
        X = solve(primal_process(x0, T, ub, lb), EM(), dt=dt)
        adjoint_process(X, luxmodel)
    end
end

function manualY(X, luxmodel)
    model, ps, st = luxmodel
    p = Lux.ComponentArray(ps)
    u(x, p, t) = model(x, p, st)[1]  # TODO: augment time to state
    y = 0.
    for (X, W, t, dt) in zip(X.u[1:end-1], X.W.W[1:end-1], X.t, diff(X.t))
        let u = u(X, p, t),
            v = v(X, t),
            f = f(X, t)
        y += (- dot(u,v) - f + 1/2 * dot(u, u) - dot(u, W)) * dt
        end
    end
    y
end

function logvar2(;ys=Yensemble(), ps=ys[1].p, sensealg=nothing)
    var(solve(y, p=ps, sensealg=sensealg)[end][1] for y in ys)
end

function dlogvar2(;ys=Yensemble(), ps=ys[1].p, sensealg=nothing)
    Zygote.gradient(ps) do p
        logvar2(;ys, ps=p, sensealg)
    end
end

function test_ad_systems()
    senses = [BacksolveAdjoint, InterpolatingAdjoint, QuadratureAdjoint,
    ReverseDiffAdjoint,
    ForwardDiffSensitivity,
    ForwardSensitivity,
    ZygoteAdjoint,
    TrackerAdjoint]
    jvs = [false, true, ZygoteVJP(), ReverseDiffVJP(true), ReverseDiffVJP(), TrackerVJP()]
    for s in senses
        for j in jvs
            try
                dlogvar2(sensealg = s(autojacvec=j))
                @time dlogvar2(sensealg = s(autojacvec=j))
                println("$s $j")
            catch e
                println("$s $j fail")
            end
        end
    end
end
