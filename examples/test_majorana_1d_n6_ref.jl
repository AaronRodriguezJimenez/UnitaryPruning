# Example usage PauliOperators
# Based on examples > plot_1d_n6_reference.jl
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators


Pbasis = PauliBasis("XYZ")
print(Pbasis)
w = PauliOperators.simple_majorana_weight(Pbasis)
println("Simple Majorana weight is: ", w)

w_two = PauliOperators.pauli_to_majorana_occupation(Pbasis)
println("Majorna weight ver 2 is: ", w_two)

function get_unitary_sequence_1D_test(o::Pauli{N}; α=.01, k=10) where N
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])
    # print("alpha", α, "\n")
    # Loop over trotter steps
    for ki in 1:k
        ## ZZ layer
        # e^{i π/2 P2} e^{i π P1 /2}|ψ>
        for i in 1:N-1
            pi = Pauli(N, Z=[i, i + 1])
            push!(generators, pi)
            push!(parameters, π/2)
        end
        #pbc 
        pi = Pauli(N, Z=[N, 1])
        push!(generators, pi)
        push!(parameters, π/2)

        ## X layer
        # e^{i αn (-X) / 2}
        for i in 1:N
            pi = Pauli(N, X=[i])
            #println("Change... Aaron was here")
            #pi = Pauli{N}((pi.θ + 2)%4, pi.pauli) # this accounts for the fact that the papers have -X and positive ZZ
            pi = Pauli{N}(-pi.s, pi.z, pi.x) #Changed due to change in Pauli type?
            push!(generators, pi)
            push!(parameters, α)
        end
    end

    return generators, parameters
end


function build_time_evolution_matrix_test(generators::Vector{Pauli{N}}, angles::Vector) where N
    U = Matrix(Pauli(N))
    nt = length(generators)
    length(angles) == nt || throw(DimensionMismatch)
    for t in 1:nt
        α = angles[t]
        U = cos(α/2) .* U .- 1im*sin(α/2) .* U * Matrix(generators[t])

    end

    return U 
end


#
#   in the experiment, the circuit is 
#
#   exp(i θ/2 (-X)) exp(i π/4 ZZ)
#

function run(;k=5, N=6)

    o = Pauli(N, Z=[1]) 
    o_mat = Matrix(o)

    e = []
    angles = []
    for i in  0:16
    # for i in [(i-1)*2 for i in 1:9]
        α=i*π/32
        
        generators, parameters = get_unitary_sequence_1D_test(o, α=α, k=k)
        U = build_time_evolution_matrix_test(generators, parameters)
        ei = diag(U'*o_mat*U)[1]

        push!(e, real(ei))
        push!(angles, α)
        
        @printf(" α: %4i val: %12.8f\n", i, ei)
    end

    plot(angles, real(e), marker = :circle)
    xlabel!("Angle")
    ylabel!("expectation value")
    savefig("plot_1d_n6_reference.pdf")
    
    return 
end

run(k=6, N=6)