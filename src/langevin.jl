using Parameters
using StochasticDiffEq
import ForwardDiff

abstract type AbstractLangevin end
# interface methods: potential(l), sigma(l), dim(l)

function SDEProblem(l::AbstractLangevin, x0=randx0(l), T=1; dt=.01, alg=SROCK2())
    drift(x,p,t) = force(l, x)
    noise(x,p,t) = sigma(l, x)
    StochasticDiffEq.SDEProblem(drift, noise, x0, T, alg=alg, dt=dt)
end

function force(l::AbstractLangevin, x)
    - ForwardDiff.gradient(x->potential(l, x), x)
end

##
@with_kw struct Doublewell <: AbstractLangevin
    dim::Int64=1
    σ::Float64=1.
end

doublewell(x) = ((x[1])^2 - 1) ^ 2

potential(::Doublewell, x) = doublewell(x)
sigma(l::Doublewell, x) = l.σ
dim(l::Doublewell) = l.dim
support(l::Doublewell) = repeat([-1.5 1.5], outer=[dim(l)])

function randx0(l::Doublewell)
    s = support(l)
    rand(size(s, 1)) .* (s[:,2] .- s[:,1]) .+ s[:,1]
end

#randx0(l::Doublewell, n) = mapslices(x->randx0(l), (1:n)', dims=1)
randx0(l::Doublewell, n) = mapreduce(x->randx0(l), hcat, 1:n)
