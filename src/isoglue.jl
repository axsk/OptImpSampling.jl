"""
reproduce behaviour of run(::Isokann)
from isokann.jl by running the newer isokann() from isonew.jl
"""
function isokann(I::AIsokann; kwargs...)
    (;nx, poweriter, learniter, dt, forcing, opt, model) = I
    nkoop = I.nmc
    usecontrol = forcing == 1
    if isa(opt, Flux.Optimise.AbstractOptimiser)
        opt = Optimisers.ADAM(0.01)
    end
    if model[end] == first
        dynamics = Doublewell()
        model = defaultmodel(dynamics, DEFAULT_LAYERS[2:end-1])
    end

    res = isokann(; nx, nkoop, poweriter, learniter, model, dt, usecontrol, opt, kwargs...)
    append!(I.ls, res.ls)
    append!(I.stds, res.stds)
    I.opt = res.opt
    I.model = model
    I
end
Optimisers.setup(opt, _) = opt  # to avoid reinitialisation of opt
