using Parameters
using StochasticDiffEq
import ForwardDiff

@with_kw struct Langevin
    V = x -> sum(abs2, x)
    σ = 1.
end

drift(lv::Langevin, x) = - ForwardDiff.gradient(lv.V, x)

function SDEProblem(lv::Langevin=Langevin(), x0=[0.], T=1; dt=.01)
    _drift(x,p,t) = drift(lv, x)
    _noise(x,p,t) = lv.σ
    StochasticDiffEq.SDEProblem(_drift, _noise, x0, T, alg=EM(), dt=dt)
end


doublewell(x) = ((x[1])^2 - 1) ^ 2

Doublewell() = Langevin(;V=doublewell)
