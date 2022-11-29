""" Comparison of the different SDE solvers,
when I remember correctly, this did not work as expected because the NoiseWrapper did not work correctly
c.f. https://github.com/SciML/StochasticDiffEq.jl/issues/505

see also
https://github.com/SciML/DiffEqNoiseProcess.jl/issues/121
https://github.com/SciML/DiffEqNoiseProcess.jl/issues/123
https://github.com/SciML/DiffEqNoiseProcess.jl/issues/126

28.11.22: looking at the issues above this could well be fixed by now.
"""

using OptImpSampling, StochasticDiffEq, DiffEqNoiseProcess, Plots

r = isokann()
sde = OptImpSampling.SDEProblem(Doublewell())
u = optcontrol(r.model, r.S, sde)
s = solve(GirsanovSDE(sde, u), save_noise=true)
W = NoiseWrapper(s.W)
s2=solve(GirsanovSDE(OptImpSampling.SDEProblem(Doublewell(), [0.], noise=W), u), save_noise=true, alg=EM(), dt=.01); plot(s2)


plot()

logspace(min, max, n) = exp.(range(log(min), log(max), n))

function plot_e_convergence!(;min=1e-4, max=1e-2, n=4, alg=EM())

    dts = logspace(min, max, n)
    e = map(dts) do  dt
        cde = GirsanovSDE(OptImpSampling.SDEProblem(Doublewell(), [0.], noise=W, alg=alg, dt=dt, abstol=dt), u)
        #plot!(s2.t, [log.(r.model(s2[1,:]')|>vec) (s2[2,:])])
        x,w = girsanovsample(cde, 0.)
        e = r.model(x)[1] * w
    end
    plot!(dts, e, xaxis=:log, label=Base.string(typeof(alg))[1:8])
end

plot()
for alg in [EM(), LambaEM(), ImplicitEM(), SROCK2()]
    try
        plot_e_convergence!(alg=alg)
    catch e
        @show e
    end
end
plot!()
