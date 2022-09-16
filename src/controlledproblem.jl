using StochasticDiffEq: is_diagonal_noise


# b + σu
function controlled_drift(D, xg, p, t; f=nothing, g=nothing, u=nothing)
    x = @view xg[1:end-1]
    ux = u(x, t)
    sigma = g(x,p,t)
    D[1:end-1] .= f(x, p, t) .+ sigma * ux
    D[end] = sum(abs2, ux) / 2
end

function controlled_noise(D, xg, p, t; g=nothing, u=nothing, diag=false)
    x = @view xg[1:end-1]
    if diag
        D[diagind(D)] .= g(x,p,t)
    else
        D[1:end-1, :] .= g(x,p,t)
    end
    D[end, 1:end] .= u(x, t)  # eq. (19)
end

ControlledSDE(l::AbstractLangevin, u) = ControlledSDE(SDEProblem(l), u)

function ControlledSDE(sde, u)
    n = length(sde.u0)
    nrp = zeros(n+1, n)
    f(D,x,p,t) = controlled_drift(D,x,p,t, f=sde.f, g=sde.g, u=u)
    g(D,x,p,t) = controlled_noise(D,x,p,t, g=sde.g, u=u, diag=is_diagonal_noise(sde))
    u0 = vcat(sde.u0, 0)

    cde = StochasticDiffEq.SDEProblem(f, g, u0, sde.tspan, sde.p; noise_rate_prototype = nrp, sde.kwargs...)
end

" convenience wrapper for obtaining X[end] and the Girsanov Weight"
function koopmansample(cde; kwargs...)
    sol=solve(cde; kwargs...)
    x = sol[end][1:end-1]
    w = exp(-sol[end][end])
    return x, w
end

function test_ControlledSDE()
    sde = SDEProblem(Doublewell())
    cde = ControlledSDE(sde, (x,t)->0.)
    koopmansample(cde, u0=[1., 2.])
end

" optcontrol(chis, q, i)

optimal control u(x,t) = -∇log(Z)
for Z = Kχᵢ if Kχ = exp(Qt) χ"
function optcontrol(chis, Q, i, sigma)
    function u(x,t)
        dlogz = Zygote.gradient(x) do x
            Z = exp(Q*t) * chis(x)
            log(Z[i])
        end
        return - sigma' * dlogz
    end
    return u
end

# TODO: check if this gives the same results as ociso
" optcontrol(x, t, chi, min, max, T)

assume χ = a1 + bϕ with Kᵀϕ = λϕ = exp(qT)
then Kᵀχ = λχ + a(1-λ)1
given minima and maxima of Kᵀχ we can estimate λ and a
and therefore compute the optimal control "
function optcontrol(x, t, chi, min, max, T, sigma)
    lambda = max-min
    a = min/(1-lambda)
    q = log(lambda) / T

    function u(x,t)
        dlogz = Zygote.gradient(x) do x
            lambda = exp(q*t)
            Z = lambda * chi(x) + a(1-λ)
            log(Z)
        end
        return - sigma' * dlogz
    end
    return u
end
