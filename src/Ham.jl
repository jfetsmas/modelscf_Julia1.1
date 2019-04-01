include("lobpcg_sep.jl")

mutable struct Ham
    Ns::Int64
    Ls::Float64
    kmul::Array{Float64,2} # wait until everything is clearer
    dx::Float64
    gridpos::Array{Float64,2}
    posstart::Int64
    posidx::Int64
    H                  # no idea what is this
    rhoa::Array{Float64,2}               # pseudo-charge for atoms (negative)
    rho::Array{Float64,2}                # electron density
    Vhar::Array{Float64,2}               # Hartree potential for both electron and nuclei
    Vtot::Array{Float64,2}               # total potential
    drhoa::Array{Float64,2}              # derivative of the pseudo-charge (on each atom)
    ev::Array{Float64,1}
    psi::Array{Float64,2}
    fermi::Float64
    occ
    nspin::Int64
    Neigs::Int64    # QUESTION: Number of eigenvalues?
    atoms::Atoms
    Eband::Float64              # Total band energy
    Fband::Float64               # Helmholtz band energy
    Ftot::Float64                # Total Helmholtz energy
    YukawaK::Float64             # shift for the potential
    epsil0::Float64
    Tbeta::Float64               # temperature 1beta = 1/T

    function Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0,Tbeta)
        Ls = Nunit * Lat;
        Ns = round(Integer, Ls / dx);

        #
        dx = Ls / Ns;
        # defining the grid
        gridpos = zeros(Ns,1) # allocating as a 2D Array
        gridpos[:,1] = transpose(collect(0:Ns-1)).*dx;
        posstart = 0;
        posidx   = 0;

        # Initialize the atom positions
        atoms   = atoms;
        Neigs = sum(atoms.nocc) + n_extra;

        Ls_glb = Ls;
        Ns_glb = Ns;

        # we define the Fourier multipliers as an 2D array
        kx = zeros(Ns,1);
        kx[:,1] = vcat( collect(0:Ns/2-1), collect( -Ns/2:-1) )* 2 * pi / Ls;
        kmul = kx.^2/2;

        rhoa, drhoa = pseudocharge(gridpos, Ls_glb, atoms,YukawaK,epsil0);

        # TODO: we need to figure out the type of each of the fields to properlu
        # initialize them
        H  = []
        rho = zeros(1,1);
        Vhar = zeros(1,1);
        Vtot = zeros(1,1);
        # drhoa = []
        ev = []
        psi = zeros(1,1);
        fermi = 0.0;
        occ = []
        Eband = 0.0
        Fband = 0.0
        Ftot = 0.0
        nspin = 1;

        new(Ns, Ls, kmul, dx, gridpos, posstart, posidx, H, rhoa,
            rho, Vhar, Vtot, drhoa, ev, psi, fermi, occ,nspin, Neigs, atoms,
            Eband, Fband, Ftot, YukawaK, epsil0, Tbeta)
    end
end

# importing the necessary functions to comply with Julia duck typing
using LinearAlgebra
import Base.*
import LinearAlgebra.mul!
import Base.eltype
import Base.size
import LinearAlgebra.issymmetric

function *(H::Ham, x::Array{Float64,1})
    # matvec overloading
    # Function  to  overload the application of the Hamiltonian times avector
    y_lap  = lap(H,x);
    y_vtot = Vtot(H,x);

    return y_lap + y_vtot;
end

# TODO: this should be optimized in order to take advantage of BLAS 3 operations
function *(H::Ham, X::Array{Float64,2})
    # Function  to  overload the application of the Hamiltonian times a matrix
    Y = zeros(size(X));
    A_mul_B!(Y, H, X)
    return Y
    # # new version that acts on the full matrices
    # Y_lap  = lap(H,X);
    # Y_vtot = Vtot(H,X);
    #
    # return Y_lap + Y_vtot;
end


function LinearAlgebra.mul!(Y::Array{Float64,2}, H::Ham, V::Array{Float64,2})
    # in place matrix matrix multiplication
    @assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end
end

# optimized version for eigs (it uses sub arrays to bypass the inference step)
function LinearAlgebra.mul!(Y::SubArray{Float64,1,Array{Float64,1}},
                  H::Ham,
                  V::SubArray{Float64,1,Array{Float64,1}})
    # in place matrix matrix multiplication
    @assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end

end

function size(H::Ham)
    return (H.Ns, H.Ns)
end

function eltype(H::Ham)
    # we work always in real space, the Fourier operations are on ly meant as
    # a pseudo spectral discretization
    return typeof(1.0)
end

function LinearAlgebra.issymmetric(H::Ham)
    return true
end


function update_psi!(H::Ham, eigOpts::eigOptions)
    # we need to add some options to the update
    # functio to solve the eigenvalue problem for a given rho and Vtot

    if eigOpts.eigmethod == "eigs"
        # TODO: make sure that eigs works with overloaded operators
        # TODO: take a look a the interface of eigs in Julia
        (ev,psi,nconv,niter,nmult,resid) = eigs(H,   # Hamiltonian
                                                nev=H.Neigs+4, # number of eigs
                                                which=:SR, # small real part
                                                ritzvec=true, # provide Ritz v
                                                tol=eigOpts.eigstol, # tolerance
                                                maxiter=eigOpts.eigsiter) #maxiter
        #assert(flag == 0);
        # TODO: this has a bug somewhere !!!! fix me!!!!
    elseif  eigOpts.eigmethod == "lobpcg_sep"
        # not working... to be fixed
        X0 = qr(rand(H.Ns, H.Neigs), thin = true)[1]
        prec(x) = inv_lap(H,x)

        (ev,psi, iter) = lobpcg_sep(H, X0, prec, H.Neigs,
                                    tol=eigOpts.eigstol,
                                    maxiter=eigOpts.eigsiter)
    elseif  eigOpts.eigmethod == "eig"
        # we use a dense diagonalization
        A = create_Hamiltonian(H)
        # checkign that A is symetric
        @assert issymmetric(A)
        (ev, psi) = (eigen(A)...,)

    end

    # sorting the eigenvalues, eigs already providesd them within a vector
    ind = sortperm(ev)[1:H.Neigs];
    # updating the eigenvalues
    H.ev = ev[ind]
    # updating the eigenvectors
    H.psi = psi[:, ind];
end



function update_rho!(H::Ham, nocc::Int64)

    ev = H.ev;
    (occ, fermi) = get_occ(ev, nocc, H.Tbeta);
    occ = occ * H.nspin;
    rho = sum(H.psi.^2*Matrix(Diagonal(occ)),dims=2)/H.dx;

    # Total energy
    E = sum(ev.*occ);

    # Helmholtz free energy
    intg = H.Tbeta*(fermi .- ev);
    ff = zeros(H.Neigs,1);
    for i = 1 : H.Neigs
      if( intg[i] > 30 )  # Avoid numerical problem.
        ff[i] = ev[i]-fermi;
      else
        ff[i] = -1/H.Tbeta * log(1+exp(intg[i]));
      end
    end
    F = sum(ff.*occ) + fermi * nocc * H.nspin;

    H.occ = occ;
    H.fermi = fermi;
    H.Eband = E;
    H.Fband = F;
    H.rho = rho;
end

function update_rhoa!(H::Ham)
    H.rhoa, H.drhoa = pseudocharge(H.gridpos, H.Ls, H.atoms,H.YukawaK,H.epsil0);
end

function create_Hamiltonian(H::Ham)
    # create the matrix version of the Hmailtonian
      A = real(ifft(diagm(0 => H.kmul[:])*fft(Matrix{Float64}(I, length(H.kmul), length(H.kmul)),1),1));
      A += diagm(0 => H.Vtot[:])
      # we symmetrize A
    return 0.5*(A + A')
end

function lap(H::Ham,x::Array{Float64,1})
    # we ask for a vector, given that we will consider the vector to be
    # a nx1 matrix
    # TODO: this can be optimized using rfft
    # TODO: we can surther accelerate this using a in-place multiplication

    ytemp = H.kmul.*fft(x);
    return real(ifft(ytemp))
end

function lap(H::Ham,x::Array{Float64,2})
    # application of the laplacian part to a matrix.
    # TODO: this can be optimized using rfft
    # TODO: we can surther accelerate this using a in-place multiplication

    ytemp = broadcast(*,H.kmul,fft(x,1));
    return real(ifft(ytemp,1))
end

function inv_lap(H::Ham,x::Array{Float64,1})
    # inverse laplacian, to be used as a preconditioner for the
    # lobpcg algorithm

    inv_kmul = zeros(size(H.kmul))
    inv_kmul[1] = 0;
    inv_kmul[2:end] = 1 ./H.kmul[2:end];

    ytemp = inv_kmul.*fft(x);
    return real(ifft(ytemp))
end

function prec(H::Ham,x::Array{Float64,1})
    # preconditioner for lobpcg
## matlab code to translate
#     for ik = 1:nkpts
#     X  = gkincell{ik};
#     Y  = 27.0 + X.*(18.0 + X.*(12.0 + 8.0*X));
#     p{ik}  = Y./(Y + 16.0*X.^4);
# end;

    inv_kmul = zeros(size(H.kmul))
    inv_kmul[1] = 0;
    inv_kmul[2:end] = 1 ./H.kmul[2:end];

    ytemp = inv_kmul.*fft(x);
    return real(ifft(ytemp))
end

function Vtot(H::Ham,x::Array{Float64,1})
    # application of the potential part of the Hamiltonian to a vector
    return (H.Vtot).*x
end

function Vtot(H::Ham, X::Array{Float64,2})
    # application of the potential part of the Hamiltonian to a matric
    return broadcast(*,H.Vtot, X)
end


function hartree_pot_bc(rho, H::Ham)
# computes the Hartree potential and energy in periodic boundary potential
# by solving the 1-d Yakuwa equation.
    # we call the (hopefully compiler optmized version)
    return hartree_pot_bc(rho, H.Ls, H.YukawaK, H.epsil0)
end




function update_vtot!(H::Ham, mixOpts)
    # TODO here we need to implement the andersson mix
    # I added the signature

    (Vtotnew,Verr) = update_pot!(H)

    # TODO: add a swtich to use different kinds of mixing here
    betamix = mixOpts.betamix;
    mixdim = mixOpts.mixdim;
    ymat = mixOpts.ymat;
    smat = mixOpts.smat;
    iter = mixOpts.iter;

    (Vtotmix,ymat,smat) = anderson_mix(H.Vtot,Vtotnew,
        betamix, ymat, smat, iter, mixdim);

    # they are already modified inside the function
    # mixOpts.ymat = ymat;
    # mixOpts.smat = smat;
    mixOpts.iter += 1;

    # updating total potential
    H.Vtot = Vtotmix;
    return Verr
end

function update_vtot!(H::Ham, mixOpts::kerkerMixOptions)
    # TODO here we need to implement the andersson mix
    # I added the signature

    (Vtotnew,Verr) = update_pot!(H)

    # println("using kerker mixing")
    # computign the residual
    res     = H.Vtot - Vtotnew;

    # println("appliying the preconditioner")
    resprec = kerker_mix(res, mixOpts.KerkerB, mixOpts.kx,
                              mixOpts.YukawaK, mixOpts.epsil0);

    # println("performing the linear mixing ")
    Vtotmix =  H.Vtot - mixOpts.betamix * resprec;

    # updating total potential
    # println("saving the total potentail ")
    H.Vtot = Vtotmix;

    # println("returning the error")
    return Verr
end

function update_vtot!(H::Ham, mixOpts::andersonPrecMixOptions)
    # TODO here we need to implement the andersson mix
    # I added the signature

    (Vtotnew,Verr) = update_pot!(H)


    betamix = mixOpts.betamix;
    mixdim = mixOpts.mixdim;
    ymat = mixOpts.ymat;
    smat = mixOpts.smat;
    iter = mixOpts.iter;

    # TODO change this so this is properly modified inside the function
    (Vtotmix,ymat,smat) = prec_anderson_mix(H.Vtot,Vtotnew,
        betamix, ymat, smat, iter, mixdim, mixOpts.prec, mixOpts.precargs)

    # they are already modified inside the function
    # just to be sure we can leave them here
    mixOpts.ymat = ymat;
    mixOpts.smat = smat;
    mixOpts.iter += 1;

    # updating total potential
    H.Vtot = Vtotmix;
    return Verr
end

import Statistics.mean
using FFTW

function init_pot!(H::Ham, nocc::Int64)
    # nocc number of occupied states
    #function to initialize the potential in the Hamiltonian class
    rho  = -H.rhoa;
    rho  = rho / ( sum(rho)*H.dx) * (nocc*H.nspin);
    H.rho = rho;
    H.Vhar = hartree_pot_bc(H.rho+H.rhoa, H);
    H.Vtot = H.Vhar;   # No exchange-correlation
    H.Vtot = H.Vtot .- mean(H.Vtot);# IMPORTANT (zero mean?)
end

function update_pot!(H::Ham)
     # TODO: I dont' know in how many different ways this si updateded
     # computing the hartree potenatial
    H.Vhar = hartree_pot_bc(H.rho+H.rhoa,H);
    # here Vtotnew only considers the
    Vtotnew  = H.Vhar;  # no exchange-correlation so far
    Verr = norm(Vtotnew-H.Vtot)./norm(H.Vtot); # computing the relative error


    # NOTE: H.Fband is only the band energy here.  The real total energy
    # is calculated using the formula below:
    H.Ftot = H.Fband + 1/2 * sum((H.rhoa-H.rho).*H.Vhar)*dx;
    return (Vtotnew,Verr) # returns the differnece betwen two consecutive iterations
end

using Arpack

# TODO: add a default setting for the scfOpts
function scf!(H::Ham, scfOpts::scfOptions)

    # vector containing the hostorical of the results
    VtoterrHist = zeros(scfOpts.scfiter)
    eigOpts = scfOpts.eigOpts;
    mixOpts = scfOpts.mixOpts;

    # number of occupied states
    Nocc = round(Integer, sum(H.atoms.nocc) / H.nspin);

    # we test first updating the psi

    for ii = 1:scfOpts.scfiter
        # solving the linear eigenvalues problem
        update_psi!(H, eigOpts);

        # update the electron density
        update_rho!(H,Nocc);

        # update the total potential, and compute the
        # differnce between the potentials
        Verr = update_vtot!(H, mixOpts);

        # save the error
        VtoterrHist[ii] = Verr ;
        # test if the problem had already satiesfied the tolerance
        if scfOpts.SCFtol > Verr
            break
        end
    end

    return VtoterrHist[VtoterrHist.>0]
end


function lap_opt(H::Ham,x::Array{Float64,1})
    # we ask for a 2 vector, given that we will consider the vector to be
    xFourier = rfft(x)
    laplacian_fourier_mult!(xFourier, H.Ls)
    return irfft(xFourier, H.Ns )
end

function lap_opt!(H::Ham,x::Array{Float64,1})
    # we ask for a 2 vector, given that we will consider the vector to be
    xFourier = rfft(x)
    laplacian_fourier_mult!(xFourier, H.Ls)
    x[:] = irfft(xFourier, H.Ns )
end

function laplacian_fourier_mult!(R::Vector{ComplexF64}, Ls::Float64 )
    c = (2 * pi / Ls)^2/2
    @simd for ii = 1:length(R)
        @inbounds R[ii] = (ii-1)^2*c*R[ii]
    end
end

# Function to compute the force and update ham.atoms.force
function get_force!(H::Ham)
    atoms  = H.atoms
    Natoms = atoms.Natoms
    rhotot = H.rho + H.rhoa
    for i=1:Natoms
        # IMPORTANT: derivative is taken w.r.t atom positions, which introduces the minus sign
        dV = -hartree_pot_bc_opt_vec(H.drhoa[:,i], H.Ls, H.YukawaK, H.epsil0)
        # Force is negative gradient
        atoms.force[i] = - sum(dV.*rhotot)*H.dx
    end
end
