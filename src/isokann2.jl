# simpler, bigger, better, stronger
# ISOKANN2 - multidimensional ISOKANN in abstract form

# todo: koopman()

function loss(xs, net, K, dynamics, n_koopman)

    fixednet = ignore_derivatives(net) # dont differentiate through the koopman estimation
    kxs = koopman(fixednet, xs, dynamics, n_koopman)
    chi = net(xs)
    K = fixk(K)

    # here we have to think about which one is actually better
    # norm(chi - K^-1 * kxs)
    norm(K * chi - kxs)
end

# softmax enforces χ-properties
net = Chain(Dense, Softmax)

# enforces rowsum = 1
function fixk(K)
    n, m = size(K)
    @assert n - 1 == m
    hcat(K, 1 .- sum(K, dims=2))
end

function train(net, K, dynamics, n_iter, n_koopman, n_startpoints)
    mod, params, state = net
    opt = Optimisers.setup(Optimisers.Adam(), params);

    for i in 1:n_iter
        xs = sample(dynamics, n_startpoints)

        l, dl = Lux.with_gradient(params, K) do (ps, K)
            net(x) = mod(x, ps, st)
            loss(xs, net, K, dynamics, n_koopman)
        end
        @show l

        opt, params = Optimisers.update(opt, params, dl)
    end

    return (mod, params, state), K
end
