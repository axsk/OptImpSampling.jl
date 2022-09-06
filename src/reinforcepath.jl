## reinforce like optimal importance sampling on the path space

# algorithm: minimisation of importance sampling variance
# with the use of girsanov reweighting and log-trick

using Parameters: @with_kw
using StatsBase: mean
using StochasticDiffEq: solve, SDEProblem, EM
using Lux: Dense, Chain
using Random: default_rng
using LinearAlgebra: dot
using ForwardDiff: gradient
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

function samplepath(x0, lv::Langevin; dt=.1)
    drift(x,p,t) = - gradient(lv.V, x)
    noise(x,p,t) = lv.σ
    prob = SDEProblem(drift, noise, x0, [0,1], save_noise=true)
    solve(prob, EM(), dt=dt)
end

function samplepath(x0, lux, lv::Langevin; dt=.1)
    (model, ps, st) = lux
    u(x) =  Lux.apply(model, x, ps, st)[1]
    drift(x,p,t) = - gradient(lv.V, x) + u(x)
    noise(x,p,t) = lv.σ
    prob = SDEProblem(drift, noise, x0, [0,1], save_noise=true)
    solve(prob, EM(), dt=dt)
end

testsamplepath() = samplepath([1.], Langevin())

function dvar_summand(Xs, lux, e::Energy = Energy())
    (model, ps, st) = lux
    u(x, ps=ps) = Lux.apply(model, x, ps, st)[1]

    Zs = map(X->W(X, e), Xs)
    dpdqs = map(X->dpdq(X, u), Xs)
    E = mean(Zs .* dpdqs)

    dV = mean(zip(Xs, Zs, dpdqs)) do (X, Z, dpdq)
        dlogq = integrate(X) do X, dW
            #-∇u(X) * (dW + u(X))

            m = :cgr
            dv = if m == :pb
                ux, ∇u = Lux.pullback(p->u(X, p), Lux.ComponentArray(ps))
                - ∇u(dW + ux)[1]
            elseif m==:cpb  # cant avoid extra computation of ux
                let u=u(X, ps)
                    pullback(dW + u, lux, X)
                end
            elseif m == :gr
                Lux.gradient( ps |> Lux.ComponentArray) do ps
                    let u=u(X, ps)
                        - dot(u, dW .+ u ./ 2)
                    end
                end[1]
            elseif m==:cgr
                mgradient(lux, X) do u
                    - dot(u, dW .+ u ./ 2)
                end
            end
            dv
        end
        #@show dlogq
        dlogq .* (E^2 - (dpdq * Z)^2)
    end
end

function testdvar()
    Xs = [testsamplepath() for i in 1:3]
    energy = Energy()
    lux = mlp(1)
    dvar_summand(Xs, lux, energy )
end

W(X, e::Energy) = let dt = diff(X.t), x = X.u[1:end-1]
    exp(-sum(e.f.(x) .* dt) + e.g(X[end]))
end

function dpdq(X, u)
    integrate(X) do X, dW
        let u = u(X)
            - u' * dW - 1/2 * u' * u
            #-dot(u, dW + u / 2)
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
    model = Chain([Dense(layers[i], layers[i+1], tanh) for i=1:length(layers)-2]...,
        Dense(layers[end-1], layers[end]))
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    model, ps, st
end

mlp(n::Int) = mlp([n, 10, n])

## lux convenience wrappers

StatefulModel = Tuple{<:Lux.AbstractExplicitLayer, <:Any, <:NamedTuple}
((mod,ps,st)::StatefulModel)(x) = mod(x,ps,st)

function mgradient(loss, (model,ps,st)::StatefulModel, x)
    Lux.gradient(ps |> Lux.ComponentArray) do ps
        loss(model(x,ps,st)[1])
    end[1]
end

function pullback(dy, (model,ps,st)::StatefulModel, x)
    u(p) = Lux.apply(model, x, p, st)[1]
    y, back = Lux.pullback(u, Lux.ComponentArray(ps))
    back(dy)[1]
end
