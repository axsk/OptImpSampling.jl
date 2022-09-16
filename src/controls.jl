using StochasticDiffEq: is_diagonal_noise

# b + σu
function controlled_drift(D, xg, p, t; f=nothing, g=nothing, u=nothing, diag=false)
    x = @view xg[1:end-1]
    ux = u(x, t)
    sigma = g(x,p,t)
    if diag
        D[1:end-1] .= f(x, p, t) .+ sigma .* ux
    else
        D[1:end-1] .= f(x, p, t) .+ sigma * ux
    end
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
    diag = is_diagonal_noise(sde)
    f(D,x,p,t) = controlled_drift(D,x,p,t, f=sde.f, g=sde.g, u=u, diag=diag)
    g(D,x,p,t) = controlled_noise(D,x,p,t, g=sde.g, u=u, diag=diag)
    u0 = vcat(sde.u0, 0)

    return StochasticDiffEq.SDEProblem(f, g, u0, sde.tspan, sde.p; noise_rate_prototype = nrp, sde.kwargs...)
end

" convenience wrapper for obtaining X[end] and the Girsanov Weight"
function girsanovsample(cde; kwargs...)
    sol=solve(cde; kwargs...)
    x = sol[end][1:end-1]
    w = exp(-sol[end][end])
    return x, w
end


" optcontrol(chis, Q, T, sigma, i)

optimal control u(x,t) = -∇log(Z)
for Z = Kχᵢ if Kχ = exp(Qt) χ"
function optcontrol(chis, Q, T, sigma, i)
    function u(x,t)
        dlogz = Zygote.gradient(x) do x
            Z = exp(Q*(T-t)) * chis(x)
            log(Z[i])
        end
        return sigma' * dlogz
    end
    return u
end

""" K on {v₁, v₂} acts like a shift-scale, represented by `Shiftscale` """
struct Shiftscale
    a::Float64
    q::Float64
end

function Shiftscale(data::AbstractArray, T=1)
    min, max = extrema(data)
    lambda = max-min
    a = min/(1-lambda)
    q = log(lambda) / T
    return Shiftscale(a, q)
end

function (s::Shiftscale)(data, T=1)
    lambda = exp(T * s.q)
    return data .* lambda .+ s.a * (1-lambda)
end

function revert(s, data, T=1)
    lambda = exp(T*s.q)
    return (data .- s.a * (1-lambda)) ./ lambda
end

# TODO: check if this gives the same results as ociso
" optcontrol(chi, kchi::Array, T, sigma)

assume χ = a1 + bϕ with Kᵀϕ = λϕ = exp(qT)
then Kᵀχ = λχ + a(1-λ)1
given minima and maxima of Kᵀχ we can estimate λ and a
and therefore compute the optimal control for Kχ = E[χ]
u* = -σᵀ∇Φ = σᵀ∇log(Kχ) "
function optcontrol(chi, kchi::Array, T, sigma)
    S = Shiftscale(kchi, T)
    optcontrol(chi, S, T, sigma)
end

function optcontrol(chi, S::Shiftscale, T, sigma)
    a, q = S.a, S.q

    function u(x,t)
        dlogz = Zygote.gradient(x) do x
            lambda = exp(q*(T-t))
            Z = lambda * chi(x) + a*(1-lambda)
            log(Z)
        end[1]
        return sigma' * dlogz
    end
    return u
end

### Tests

function test_ControlledSDE()
    sde = SDEProblem(Doublewell())
    cde = ControlledSDE(sde, (x,t)->0.)
    koopmansample(cde, u0=[1., 2.])
end

function test_compare_controls()
    chi(x) = (sin(x[1]) + 1.1) / 3
    xs = [[1.], [0.], [-.1]]
    x = xs[1]
    T = 1.

    q = -1.
    b = 0.
    o0 = ProblemOptChi(chi=chi, q=q, b=b, forcing=0, T=T)

    kxs, std, λ, b, ys = SK(o0, xs, 3)
    q = log(λ)

    cs = map(chi, ys)
    x = xs[1]

    o1 = ProblemOptChi(chi=chi, q=q, b=b, forcing=1, T=T)
    c1 = control(o1, x, T)
    c2 = optcontrol(chi, cs, T, o1.σ)(x, T)
    @assert c1 == c2
end
