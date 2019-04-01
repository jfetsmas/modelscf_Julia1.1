
struct eigOptions
    # structure to encapsulate all the different options for the scf iterations
    # the main idea is to define a Hamiltonian, and then run scf on it; however,
    # the Hamiltoinian shouldn't have the scf options embeded
    eigstol::Float64
    eigsiter::Int64
    eigmethod::AbstractString

    function eigOptions()
        new(1e-8, 100, "eigs")
    end

    function eigOptions(eigstol::Float64, eigsiter::Int64, eigmethod::String)
        new(eigstol, eigsiter, eigmethod)
    end
end

# defining abstract type for all the different mixing options
abstract type mixingOptions end

mutable struct simpleMixOptions <: mixingOptions
    betamix::Float64
    function simpleMixOptions(betamix)
        new(betamix)
    end
    # default constructor
    function simpleMixOptions()
        new(0.5)
    end
end

mutable struct andersonMixOptions <: mixingOptions
    ymat::Array{Float64,2}
    smat::Array{Float64,2}
    betamix::Float64
    mixdim::Int64
    iter::Int64
    function andersonMixOptions(Ns, betamix, mixdim )
        ymat = zeros(Ns, mixdim);
        smat = zeros(Ns, mixdim);
        new(ymat,smat,betamix[1],mixdim,1)
    end
end

mutable struct andersonPrecMixOptions <: mixingOptions
    ymat::Array{Float64,2}
    smat::Array{Float64,2}
    betamix::Float64
    mixdim::Int64
    iter::Int64
    prec::Function
    precArgs
    function andersonPrecMixOptions(Ns, betamix, mixdim, prec, precArgs)
        ymat = zeros(Ns, scfOpts.mixdim);
        smat = zeros(Ns, scfOpts.mixdim);
        new(ymat,smat,betamix[1],mixdim,1, prec, precArgs)
    end
end


mutable struct kerkerMixOptions <: mixingOptions
    betamix::Float64
    KerkerB::Float64
    kx::Array{Float64,2}
    YukawaK::Float64
    epsil0::Float64
    function kerkerMixOptions(betamix,KerkerB, kx, YukawaK, epsil0)
        new(betamix,KerkerB, kx, YukawaK, epsil0)
    end
end

function updateMix!( mixOpts::andersonMixOptions, ii )
    mixOpts.iter = ii;
end

struct scfOptions
    # structure to encapsulate all the different options for the scf iterations
    # the main idea is to define a Hamiltonian, and then run scf on it; however,
    # the Hamiltoinian shouldn't have the scf options embeded
    SCFtol::Float64
    scfiter::Int64
    eigOpts::eigOptions
    mixOpts::mixingOptions

    # TODO: this Initializing is ugly, we need to use kwargs to properly initialize
    function scfOptions()
        eigOpts = eigOptions();
        mixOpts = simpleMixOptions();
        new(1e-7,100, eigOpts, mixOpts);
    end
    function scfOptions(SCFtol::Float64,scfiter::Int64,
                        eigOpts::eigOptions,
                        mixOpts::mixingOptions)
        new(SCFtol,scfiter, eigOpts, mixOpts)
    end
    function scfOptions(eigOpts::eigOptions, mixOpts::mixingOptions)
        new(1e-7,100,  eigOpts, mixOpts)
    end

end
