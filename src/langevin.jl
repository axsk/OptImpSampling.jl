using Parameters
using StochasticDiffEq
import ForwardDiff

abstract type AbstractLangevin end
# interface methods: potential(l), sigma(l)
force(l::AbstractLangevin, x) = - ForwardDiff.gradient(potential(l), x)

@with_kw struct Langevin <: AbstractLangevin
    V = x -> sum(abs2, x)
    σ = 1.
end

potential(l::Langevin) = l.V
sigma(l::Langevin) = l.σ

function SDEProblem(l::AbstractLangevin, x0=randx0(l), T=1; dt=.01)
    _drift(x,p,t) = force(l, x)
    _noise(x,p,t) = sigma(l)
    StochasticDiffEq.SDEProblem(_drift, _noise, x0, T, alg=EM(), dt=dt)
end



@with_kw struct Doublewell <: AbstractLangevin
    dim=1
    σ=1.
end

doublewell(x) = ((x[1])^2 - 1) ^ 2

potential(::Doublewell) = doublewell
sigma(l::Doublewell) = l.σ
dim(l::Doublewell) = l.dim
support(l::Doublewell) = repeat([-1.5 1.5], outer=[dim(l)])

function randx0(l::Doublewell)
    s = support(l)
    rand(size(s, 1)) .* (s[:,2] .- s[:,1]) .+ s[:,1]
end

randx0(l::Doublewell, n) = mapslices(x->randx0(l), (1:n)', dims=1)
