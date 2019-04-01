# script to obtain the oscillating atoms using a vanilla verlet Integrator
# The main idea is to build a reference model to study the behavior fo the
# Neural network accelerated Molecular dynamic.

# we use the modelscf to compute the forces, and the velocity verlet algorithm
# to evolve the system.

# in this case we save the evolution of the system in a hd5f file.

include("../src/Atoms.jl")
include("../src/scfOptions.jl")
include("../src/anderson_mix.jl")
include("../src/kerker_mix.jl")
include("../src/Ham.jl")
include("../src/hartree_pot_bc.jl")
include("../src/pseudocharge.jl")
include("../src/getocc.jl")
include("../src/Integrators.jl")
using HDF5

# flag to save the data

# number of MD simulations and
Nit = 3000

# getting all the parameters
dx = 0.5;
Nunit = 8;   # number of units
Lat = 10;     # size of the lattice
Ls = Nunit*Lat;
Ns = round(Integer, Ls / dx); # number of discretization points

# using the default values in Lin's code
YukawaK = 0.0100
n_extra = 10;
epsil0 = 10.0;
T_elec = 100.0;

kb = 3.1668e-6;
au2K = 315774.67;
Tbeta = au2K / T_elec;

betamix = 0.5;
mixdim = 10;

Ndist = 1;
Natoms = round(Integer, Nunit / Ndist);

sigma  = ones(Natoms,1)*(1.0);  # insulator
omega  = ones(Natoms,1)*0.03;
Eqdist = ones(Natoms,1)*10.0;
mass   = ones(Natoms,1)*42000.0;
nocc   = ones(Natoms,1)*2;          # number of electrons per atom
Z      = nocc;

function forces(x::Array{Float64,1})
    # input
    #       x: vector with the position of the atoms
    # output
    #       f: forces at the center of the atoms

    R = reshape(x, length(x), 1) # we need to make it a two dimensional array
    # creating an atom structure
    atoms = Atoms(Natoms, R, sigma,  omega,  Eqdist, mass, Z, nocc);
    # allocating a Hamiltonian
    ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta)

    # total number of occupied orbitals
    Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);

    # setting the options for the scf iteration
    mixOpts = andersonMixOptions(ham.Ns, betamix, mixdim )
    eigOpts = eigOptions(1.e-10, 1000, "eig");
    scfOpts = scfOptions(1.e-8, 300, eigOpts, mixOpts)

    # initialize the potentials within the Hemiltonian, setting H[\rho_0]
    init_pot!(ham, Nocc)

    # running the scf iteration
    VtoterrHist = scf!(ham, scfOpts)

    if VtoterrHist[end] > scfOpts.SCFtol
        println("convergence not achieved!! ")
    end

    # we compute the forces
    get_force!(ham)

    # computing the energy
    Vhar = hartree_pot_bc(ham.rho+ham.rhoa,ham);

    # NOTE: ham.Fband is only the band energy here.  The real total energy
    # is calculated using the formula below:
    Etot = ham.Eband + 1/2*sum((ham.rhoa-ham.rho).*Vhar)*dx;

    return ham.atoms.force[:]
end

# Setting the time evolution
dt = 0.01

x0 = zeros(Natoms); # this is defined as an 2D array
for j = 1:Natoms
  x0[j] = (j-0.5)*Lat*Ndist+dx;
end

x0[1] = x0[1] + 1.0

v0 = zeros(Natoms)
x1 = x0 + dt*v0

#### Computing the trajectories

(X_traj, v, vdot) = time_evolution(velocity_verlet, x -> forces(x), dt, Nit, x0, x1)

Rho    = zeros(Ns, Nit)
Etotal = zeros(1, Nit)
Ftotal = zeros(Natoms, Nit)


# we re-run the DFT computation using the trajectory
for ii = 1:Nit

    # we use the data of the evolved function
    R = reshape(X_traj[ii, :],Natoms,1)

    # creating an atom structure
    atoms = Atoms(Natoms, R, sigma,  omega,  Eqdist, mass, Z, nocc);

    # allocating a Hamiltonian
    ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta)

    # total number of occupied orbitals
    Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);

    # initialize the potentials within the Hemiltonian, setting H[\rho_0]
    init_pot!(ham, Nocc)

    # we use the anderson mixing of the potential
    mixOpts = andersonMixOptions(ham.Ns, betamix, mixdim )

    # we use the default options
    eigOpts = eigOptions(1.e-12, 1000, "eig");

    scfOpts = scfOptions(1.e-10, 3000, eigOpts, mixOpts)

    # running the scf iteration
    @time VtoterrHist = scf!(ham, scfOpts)

    # we compute the forces
    get_force!(ham)

    # computing the energy
    Vhar = hartree_pot_bc(ham.rho+ham.rhoa,ham);

    # NOTE: ham.Fband is only the band energy here.  The real total energy
    # is calculated using the formula below:
    Etot = ham.Eband + 1/2*sum((ham.rhoa-ham.rho).*Vhar)*dx;

    Etotal[ii] = Etot
    Ftotal[:,ii] = ham.atoms.force[:]
    Rho[:,ii] = ham.rho[:];

    println(length(VtoterrHist))
end

print("simulation finished")
Input_str = string("../training/data/Input_KS_MD_scf_", Natoms,"_sigma_", sigma[1],".h5")
Output_str = string("../training/data/Output_KS_MD_scf_", Natoms,"_sigma_", sigma[1],".h5")

# we remoce the files if they are still around
isfile(Output_str) && rm(Output_str)
isfile(Input_str)  && rm(Input_str)

h5write(Input_str, "R", X_traj[1:Nit,:])

h5open(Output_str, "w") do file
    write(file,"Rho", Rho) # alternatively, say "@write file A"
    write(file,"Etotal", Etotal)
    write(file,"Ftotal", Ftotal)
end