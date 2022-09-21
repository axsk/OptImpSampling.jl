using StochasticDiffEq: is_diagonal_noise
using FLoops: @floop

# b + σu
function controlled_drift(D, xg, p, t;
        f::F=nothing, g::G=nothing, u::H=nothing, diag=false) where {F,G,H}
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

function controlled_noise(D, xg, p, t;
        g::G=nothing, u::H=nothing, diag=false) where {G,H}
    x = @view xg[1:end-1]
    D .= 0
    if diag
        D[diagind(D)[1:end-1]] .= g(x,p,t)
    else
        D[1:end-1, 1:end-1] .= g(x,p,t)
    end
    D[end, 1:end-1] .= u(x, t)  # eq. (19)
end

function ControlledSDE(sde, u::F) where F
    n = length(sde.u0)
    nrp = zeros(n+1, n+1)
    u0 = vcat(sde.u0, 0)

    #diag = is_diagonal_noise(sde)
    # f(D,x,p,t) = controlled_drift(D,x,p,t, f=sde.f, g=sde.g, u=u, diag=diag)
    # g(D,x,p,t) = controlled_noise(D,x,p,t, g=sde.g, u=u, diag=diag)

    function f(D,x,p,t)
        x = SVector{1}(@view x[1:end-1])

        fx = sde.f(x,p,t)
        gx = sde.g(x,p,t)
        ux = u(x,t)
        D[1:end-1] .= fx .+ gx .* ux
        D[end] = sum(abs2, ux)
    end
    function g(D,x,p,t)
        x = SVector{1}(@view x[1:end-1])
        gx = sde.g(x,p,t)
        ux = u(x,t)
        D .= 0
        for i in eachindex(x)
            D[i,i] = gx
        end
        D[end, 1:end-1] .= ux
    end

    return StochasticDiffEq.SDEProblem(f, g, u0, sde.tspan, sde.p;
        noise_rate_prototype = nrp, sde.kwargs...)
end

nocontrol(x, t) = zero(x)

" convenience wrapper for obtaining X[end] and the Girsanov Weight"
function girsanovsample(cde, x0; kwargs...)
    u0 = vcat(x0, 0)
    sol=solve(cde; u0=u0, kwargs...)
    x = sol[end][1:end-1]
    w = exp(-sol[end][end])
    return x, w
end

# TODO: maybe use DiffEq MC interface
function girsanovbatch(cde, xs, n)
    dim, nx = size(xs)
    ys = zeros(dim, nx, n)
    ws = zeros(nx, n)
    @floop for i in 1:nx, j in 1:n
            ys[:, i, j], ws[i, j] = girsanovsample(cde, xs[:, i])
    end
    return ys, ws
end

" optcontrol(chis, Q, T, sigma, i)

optimal control u(x,t) = -∇log(Z)
for Z = Kχᵢ if Kχ = exp(Qt) χ.
Given it terms of the known generator Q"
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
    a, b = extrema(data)
    lambda = b-a
    a = a/(1-lambda)
    q = log(lambda) / T
    return Shiftscale(a, q)
end

function (s::Shiftscale)(data, T=1)
    lambda = exp(T * s.q)
    return data .* lambda .+ s.a * (1-lambda)
end

function invert(s, data, T=1)
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

function optcontrol(chi::F, S::Shiftscale, T, sigma) where F
    function u(x,t)
        #x = SVector{length(x)}(x)
        dlogz = fgrad(x) do x
            lambda = exp(S.q*(T-t))
            Z = lambda * first(chi(x)) + S.a*(1-lambda)
            log(Z)
        end #:: Vector{Float64}  # TODO: this should be inferred!
        return sigma' * dlogz
    end
    return u
end

# convenience wrapper using the original sde to extract noise and T
function optcontrol(model, S::Shiftscale, sde)
    sigma = sde.g(nothing, nothing, nothing)
    T = sde.tspan[end]
    optcontrol(model, S, T, sigma)
end

### Tests

function test_ControlledSDE()
    sde = SDEProblem(Doublewell())
    cde = ControlledSDE(sde, nocontrol)
    ys, ws = girsanovbatch(cde, rand(1,2), 3)
end

function test_optcontrol()
    sde = SDEProblem(Doublewell())
    model = densenet([1,3,3,1])
    u = optcontrol(model, Shiftscale(1,0), 1, 1)
    cde = ControlledSDE(sde, u)
    ys, ws = girsanovbatch(cde, rand(1,2), 3)
end

function test_compare_controls()
    chi(x) = (sin(x[1]) + 1.1) / 3
    xs = [[1.], [0.], [-.1]]
    x = xs[1]
    T = 1.

    q = -1.
    b = 0.

    # generate samples and estimate old way
    o0 = ProblemOptChi(chi=chi, q=q, b=b, forcing=0, T=T)
    kxs, std, λ, b, ys = SK(o0, xs, 3)
    q = log(λ)

    cs = map(chi, ys)
    x = xs[1]

    o1 = ProblemOptChi(chi=chi, q=q, b=b, forcing=1, T=T)
    c1 = control(o1, x, T)

    # new way
    S = Shiftscale(cs, T)
    u2 = optcontrol(chi, S, T, o1.σ)
    @show c2 = u2(x, T)
    @assert c1 == c2


    s1 = SDEProblem(o1, x)
    s2 = ControlledSDE(SDEProblem(Doublewell()), u2)

    d0 = copy(x)
    @show s2.f(d0, x, nothing, 0)
    @show d0

    du = copy(x)
    @show controlled_drift(du, x, o1, 0)
    @show du
end
