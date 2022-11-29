module OptImpSampling

include("utils.jl")     # neural network convenience

include("langevin.jl")  # langevin process
include("control.jl")   # opt control

include("humboldtsample.jl")  # adaptive sampling
include("isonew.jl")    # new implementation of isokann

export isokann


#include("sqra.jl")     # copy from Sqra.jl for reference solution
#include("ociso.jl")    # old opt control
#include("isokann.jl")  # old implementation of isokann

#include("isokann2.jl") # experimental implementation of new isokann arch
#include("dynamics.jl")  # experimental reconstruction of dynamcis from kinetics

#include("fbsde.jl)     # my fbsde idea
#include("reinforce.jl")  # my reinforce like (state) importance sampling
#include("reinforcepath.jl")  # adaptation of above to path space

#include("logvar.jl")   # lorenz' logvar optimization

end
