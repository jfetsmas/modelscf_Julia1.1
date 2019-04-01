# Functions implementing the time evolution
# We only implement (which seems to work) the velocity verlet algorithm. 

#TODO: implement the Extended Lagrangian based methods and integrators

function  velocity_verlet(a::Function,   # function providing the acceleration
                                            x::Array{T,1}, # current position
                                            v::Array{T,1}, # current speed
                                            vdot::Array{T,1}, # current acceleration
                                            h::T) where T <: Number           # delta time
    
# modification to have only one evaluation of a per iteration
    v_half = v      + h/2 * vdot
    x_next = x      + h   * v_half
    vdot_next = a(x_next)
    v_next = v_half + h/2 * vdot_next
    return (x_next, v_next, vdot_next)
end

function time_evolution(Integrator::Function, # integrator 
                                          a::Function,          # acceleration
                                          h::T,                 # time step
                                          Nsteps::Int64,        # number of stepd for evolution
                                          x0::Array{T,1},       # initial guess
                                          x1::Array{T,1} ) where T <: Number      # initial+1 guess

xArray = Array{T,2}(undef, Nsteps+1, length(x0))
vArray = Array{T,2}(undef, Nsteps+1, length(x0))
aArray = Array{T,2}(undef, Nsteps+1, length(x0))

println("initial conditions")
xArray[1,:] =  x0
xArray[2,:] =  x1

vArray[1,:] = (x1 - x0)/(h) # is this enough?
aArray[1,:] = a(x0)

println("running the loop")

for ii = 1:Nsteps
    (x, v, vdot) = Integrator(a,xArray[ii,:], vArray[ii,:], aArray[ii,:], h)
    xArray[ii+1,:] = x
    vArray[ii+1,:] = v
    aArray[ii+1,:] = vdot
end
println("loop finished")

return (xArray, vArray, aArray)
end



