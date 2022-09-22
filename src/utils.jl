import Flux
import Lux

DEFAULT_LAYERS = [1,3,3,1]


### FLUX
function fluxnet(layers=DEFAULT_LAYERS, act = Flux.sigmoid, lastact=act)
    Flux.Chain(
        [Flux.Dense(layers[i], layers[i+1], act) for i in 1:length(layers)-2]...,
        Flux.Dense(layers[end-1], layers[end], lastact))
end

# we use this to create a copy which uses StaticArrays, for faster d/dx gradients
statify(x::Any) = x
statify(c::Flux.Chain) = Flux.Chain(map(statify, c.layers)...)
function statify(d::Flux.Dense)
    w = d.weight
    W = SMatrix{size(w)...}(w)
    b = d.bias
    B = SVector{length(b)}(b)
    Flux.Dense(W, B, d.σ)
end


### LUX

# multi layer perceptron for the force
function luxnet(layers=DEFAULT_LAYERS, act = Lux.sigmoid, lastact=act)
    model = Lux.Chain(
        [Lux.Dense(layers[i], layers[i+1], act) for i in 1:length(layers)-2]...,
        Lux.Dense(layers[end-1], layers[end], lastact))
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    model, ps, st
end

StatefulModel = Tuple{<:Lux.AbstractExplicitLayer, <:Any, <:NamedTuple}
((mod,ps,st)::StatefulModel)(x) = mod(x,ps,st)[1]

# I believe this is the natural way to interpret the gradient of a loss wrt the model
function Lux.gradient(loss, (model,ps,st)::StatefulModel, x)
    Lux.gradient(ps |> Lux.ComponentArray) do ps
        loss(model(x,ps,st)[1])
    end[1]
end

# TODO: this needs testing before we open a PR to Lux
function pullback(dy, (model,ps,st)::StatefulModel, x)
    u(p) = Lux.apply(model, x, p, st)[1]
    y, back = Lux.pullback(u, Lux.ComponentArray(ps))
    back(dy)[1]
end


# SIMPLECHAINS: thought about it, but they dont provide derivatives wrt. x
