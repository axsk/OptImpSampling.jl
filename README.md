# OptImpSampling.jl

A WIP collection of algorithms related to optimal importance sampling (OIS).

With a focus on ISOKANN:
- [ociso](src/ociso.jl): OIS for Koopman eigen- and chi functions
- [isokann](src/isokann.jl): ISOKANN using OIS for the chi approximations

- [control](src/control.jl): simpler rewrite of ociso
- [isonew](src/isonew.jl): rewrite of isokann, uses humboldtsampling

- [isokann2](src/isokann2.jl): sketch of ISOKANN in higher dimensions

Utils:
- [sqra](src/sqra.jl): Square root approximation for the generators of Ito diffusions (copy from Sqra.jl)
- [humboldtsample](src/humboldtsample.jl): sample uniform along the reaction coordinate

Experimental implementations:
- [logvar](src/logvar.jl): OIS  of path functionals with control variates (implements Richter, Nusken 2021)
- [reinforce](src/reinforce.jl): OIS of functions via REINFORCE
- [reinforcepath](src/reinforcepath.jl): like reinfore but on path space (equivalent to logvar)
- [fbsde](src/fbsde.jl): Forward backward SDE with Neural Controls, learning all times at once (like Kebiri, Hartmann 2019)
