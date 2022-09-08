## reinforce like optimal importance sampling on the path space

# algorithm: minimisation of importance sampling variance
# with the use of girsanov reweighting and log-trick

using Parameters: @with_kw
using StatsBase: mean
using DiffEqBase: ContinuousCallback
using StochasticDiffEq: solve, SDEProblem, EM
using Lux: Dense, Chain
using Random: default_rng
using LinearAlgebra: dot
import ForwardDiff
import Random
import Lux


@with_kw struct Energy
    f = x -> 1
    g = x -> 0
    stoptime = x -> x[1] - 1  # not implemented yet
end

@with_kw struct Langevin
    V = x -> sum(abs2, x)
    σ = 1.
end

drift(lv::Langevin, x) = - ForwardDiff.gradient(lv.V, x)

function samplepath(x0, lv::Langevin=Langevin(), u=(x->zero(x0)); dt=.1)
    _drift(x,p,t) = drift(lv, x) + lv.σ * u(x)
    _noise(x,p,t) = lv.σ
    termcond = ContinuousCallback((u,t,int)->lv.stoptime(u), terminate!)
    prob = SDEProblem(drift, noise, x0, [0,1], save_noise=true, callback=termcond)
    solve(prob, EM(), dt=dt)
end

testsamplepath() = samplepath([1.], Langevin())

function dvar(Xs, u::StatefulModel, e::Energy = Energy(), lv = Langevin())
    Zs = W.(Xs, Ref(e))  # Ref(e) means don't broadcast over e
    dpdqs = dpdq.(Xs, Ref(u))
    E = mean(Zs .* dpdqs)
    σ⁻ = inv(lv.σ)

    dV = mean(zip(Xs, Zs, dpdqs)) do (X, Z, dpdq)
        dlogq = integrate(X) do X, dW
            # convenince wrapper for gradient of loss(u(X)) wrt params of lux
            Lux.gradient(u, X) do uₓ
               ũ = uₓ + σ⁻ * drift(lv, X)
               - ũ'dW - ũ'ũ / 2
            end
        end
        dlogq .* (E^2 - (dpdq * Z)^2)
    end
end

function testdvar()
    Xs = [testsamplepath() for i in 1:3]
    energy = Energy()
    lux = mlp(1)
    dvar(Xs, lux, energy)
end

W(X, e::Energy) = exp(-integrate((x, dW) -> e.f(x), X) + e.g(X[end]))

function dpdq(X, u)
    integrate(X) do X, dW
        let u = u(X)
            - u' * dW - 1/2 * u' * u
        end
    end
end

# integrate a stochastic function f(X, dW) along the path X: ∫ f(X(t), dW(t)) dt
function integrate(f, X)
    sum(zip(X.u, diff(X.t), diff(X.W.u))) do (X, dt, dW)
        f(X, dW) .* dt
    end
end

# multi layer perceptron for the force
function mlp(layers=[1,10,1])
    model = Lux.Chain(
        [Lux.Dense(layers[i], layers[i+1], tanh) for i=1:length(layers)-2]...,
        Lux.Dense(layers[end-1], layers[end]))
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    model, ps, st
end

mlp(n::Int) = mlp([n, 10, n])

## lux convenience wrappers

StatefulModel = Tuple{<:Lux.AbstractExplicitLayer, <:Any, <:NamedTuple}
((mod,ps,st)::StatefulModel)(x) = mod(x,ps,st)[1]

function Lux.gradient(loss, (model,ps,st)::StatefulModel, x)
    Lux.gradient(ps |> Lux.ComponentArray) do ps
        loss(model(x,ps,st)[1])
    end[1]
end

function pullback(dy, (model,ps,st)::StatefulModel, x)
    u(p) = Lux.apply(model, x, p, st)[1]
    y, back = Lux.pullback(u, Lux.ComponentArray(ps))
    back(dy)[1]
end
