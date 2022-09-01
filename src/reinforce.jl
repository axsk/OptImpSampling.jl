# proof of concept of the REINFORCE-like algorithm
# to learn the optimal importance sampler Z = E_p[f] = E_q[pf/q]
# i.e. the optimal proposal density q* = pf / Z
# for pointwise (not pathwise) expectations
# works so far

using Plots
using Distributions
using ForwardDiff

test() = train()

###

# mc estimation of the variance of the importance sampling estimator
function dtheta!(dtheta, Q, p, f, E, samples)
    dtheta .= 0. # derivative
    e = 0. # expectation value
    var = 0. # empirical variance

    for i in 1:samples
        x = rand(Q)
        qx = pdf(Q, x)
        dqx = dpdf(Q, x)
        px = p(x)
        fx = f(x)

        # TODO: write down the derivation
        dtheta .+= dqx / qx * (E^2 - (px*fx/qx)^2) # this is where the meat is
        e += fx * px / qx
        var += (E-(px*fx/qx))^2
    end

    dtheta ./= samples
    e = e / samples
    var = var / samples

    return e
end

doublewell(x) = ((x[1])^2 - 1) ^ 2
stationary(f) = x->exp(-f(x)) # only up to normalizing constant

function train(f=stationary(doublewell); Q=GaussMixture(), steps=1000, alpha=0.05, p=x->1, samples=10)

    theta = extractparms(Q)
    dtheta = similar(theta)

    es = []
    plot(f, label="target")
    plot!(x->pdf(Q, x), label="start")
    E = 0
    for i in 1:steps
        e = dtheta!(dtheta, Q, p, f, E, samples)
        theta -= dtheta * alpha

        E = E * (1 - 1/i) + e/i
        push!(es, e)

        Q = GaussMixture(theta)
        #plot(f);
        #plot!(x->pdf(Q, x)) |> display;
    end
    pl = plot!(x->pdf(Q, x), label="fit")

    #plot(es, yaxis="E") |> display
    display(pl)
    @show E
    return Q
end

###

# For starters, use a Mixture of Gaussians
# we will want to replace this with invertible NN later on

using Distributions
function GaussMixture(theta = randn(15))
    theta = reshape(theta, 3, :)
    normals = Normal[]
    w = []
    for i in 1:Int(length(theta)/3)
        push!(normals, Normal(theta[1,i], exp(theta[2,i])))
        push!(w, exp(theta[3,i]))
    end
    #ms = p[1:2, :] |> eachcol .|> collect .|> x->tuple(x...)
    w = w / sum(w)

    return MixtureModel(normals, w)
end

dpdf(Q, x) = ForwardDiff.gradient(q->pdf(GaussMixture(q), x), extractparms(Q))

# extract the parameters mean, std, and weiht of the gaussian
function extractparms(q)
    c = components(q)
    p = zeros(3, length(c))
    for i in 1:length(c)
        p[1:2,i] .= params(c[i])
    end
    p[2,:] .= log.(p[2,:])
    p[3,:] .= log.(probs(q))
    return p
end
