#
# SYSTEMATIC STUDY OF TFIM Model
# Loops over the desired combinations of N and k
# For each combination, evaluates results for both weight strategies
# Plots and save the results for direct comparison

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
    α = 1.0   # π / 32 
    
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

############### Generate results functions #######################
function plot_series_and_save_results(; 
    Ns = [4, 6, 8, 10], 
    ks = [6, 8, 10], 
    output_dir = "TFIM_series_results",
    txt_filename = "data_1d_bfs_fim_series.txt"
)

    # Make directory if it doesn't exist
    isdir(output_dir) || mkpath(output_dir)

    txt_file_path = joinpath(output_dir, txt_filename)
    open(txt_file_path, "w") do io
        for N in Ns
            for k in ks
                println("Running N = $N, k = $k")
                max_weights = 1:1:N #Max weight not larger than size of the qubit model
                errors_dict = Dict("Pauli" => Float64[], "Majorana" => Float64[])

                for w_type in ["Pauli", "Majorana"]
                    for mw in max_weights
                        println("  -> w_type = $w_type, max_weight = $mw")
                        err = run(N=N, k=k, w_type=w_type, max_weight=mw)
                        push!(errors_dict[w_type], err)
                        # Save result line to file
                        @printf(io, "N=%d k=%d w_type=%s max_weight=%d error=%.6e\n", N, k, w_type, mw, err)
                    end
                end

                # Plot comparison
                plt = plot(
                    max_weights, errors_dict["Pauli"],
                    label = "Pauli",
                    marker = :circle,
                    lw = 2
                )
                plot!(
                    max_weights, errors_dict["Majorana"],
                    label = "Majorana",
                    marker = :square,
                    lw = 2
                )
                title!("Error vs Max Weight (N=$N, k=$k)")
                xlabel!("Max Weight Cutoff")
                ylabel!("Absolute Error")
                

                filename = "abs_error_vs_weight_N=$N-k=$k.pdf"
                savefig(plt, joinpath(output_dir, filename))
                println("Saved plot as $filename")
            end
        end
    end

    println("All results saved in $txt_file_path")
end

# Call the function to execute the study
plot_series_and_save_results()