#https://github.com/FluxML/Zygote.jl/issues/1294

using Zygote
using StochasticDiffEq, SciMLSensitivity
import Lux

function mwe()
    x0 = rand(1)
    p0 = rand(1)

    drift(du,u,p,t) = (du .= 1)
    noise(du,u,p,t) = (du .= 1)

    prob = SDEProblem(drift, noise, x0, 1., p0)
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())
    Zygote.gradient(p0) do p
        sum(Zygote.@showgrad(solve(remake(prob, p=p), EM(), dt=.1, sensealg=sensealg)[end][1]) for i in 1:3)
    end
end
mwe()
