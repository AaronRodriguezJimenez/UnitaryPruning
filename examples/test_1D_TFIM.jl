using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators

#
# - - - 1D Transverse Field Ising Model (TFIM)
#
function get_unitary_sequence_1D_test(o::Pauli{N}; α=.01, k=10) where N
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])

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
            pi = Pauli{N}(-pi.s, pi.z, pi.x) # this accounts for the fact that the papers have -X and positive ZZ
            push!(generators, pi)
            push!(parameters, α)
        end
    end

    return generators, parameters
end


#
#   in the experiment, the circuit is 
#
#   exp(i θ/2 (-X)) exp(i π/4 ZZ)
#

function run(; N=10, k=6, w_type = "Majorana", max_weight=1)
   
    ket = Ket(N, 0) 
    o = Pauli(N, Z=[1])
    α = 1# π / 32 
    
    # Generators and parameters for a single angle
    generators, parameters = get_unitary_sequence_1D_test(o, α=α, k=k)

    ei, nops = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)

    # Exact evolution
    U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    o_mat = Matrix(o)
    m = diag(U'*o_mat*U)
    abs_err = abs(real(m[1])- real(ei) )
    println("Exact :", real(m[1]), " Approx :", real(ei), " Absolute Error: ", abs_err)
    return abs_err
end

function plot_abs_error_vs_weight_pdf(; N=6, k=10, w_type="Pauli", max_weights=0:2:6)
    errors = Float64[]
    weights = Int[]
    
    for mw in max_weights
        println("Evaluating max_weight = $mw")
        err = run(N=N, k=k, w_type=w_type, max_weight=mw)
        push!(errors, err)
        push!(weights, mw)
    end

    plt = plot(
        weights, errors,
        xlabel = "Max Weight Cutoff",
        ylabel = "Absolute Error",
        title = "Error vs Max Weight Cutoff (N = $N, k = $k)",
        marker = :circle,
        lw = 2,
        legend = false,
        grid = true
    )

    filename="1D_TFIM_abs_error_vs_weight_$w_type-N=$N-k=$k.pdf"
    savefig(plt, filename)
    println("Plot saved as $filename")
end
    
plot_abs_error_vs_weight_pdf(N=6, k=10, w_type="Pauli", max_weights=1:2:10)