#
# SYSTEMATIC STUDY OF HEISENBERG Model
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


"""
 1D linear chain verison of the Heisenberg model
"""
function heisenberg_1D(o::Pauli{N}; Jx, Jy, Jz, k) where N 
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])
    # Loop over sites
    for ki in 1:k 
        for i in 1:N-1
            push!(generators, Pauli(N, X=[i, i + 1]))
            push!(parameters, Jx)
            push!(generators, Pauli(N, Y=[i, i + 1]))
            push!(parameters, Jy)
            push!(generators, Pauli(N, Z=[i, i + 1]))
            push!(parameters, Jz)
        end
    end

    return generators, parameters
end

function heisenberg_1D(N, Jx, Jy, Jz; x=0, y=0, z=0)
    H = PauliSum(N, Float64)
    for i in 0:N-1
        H += -2*Jx * Pauli(N, X=[i+1,(i+1)%(N)+1])
        H += -2*Jy * Pauli(N, Y=[i+1,(i+1)%(N)+1])
        H += -2*Jz * Pauli(N, Z=[i+1,(i+1)%(N)+1])
    end 
    for i in 1:N
        H += x * Pauli(N, X=[i])
        H += y * Pauli(N, Y=[i])
        H += z * Pauli(N, Z=[i])
    end 
    return H
end

function run(; N=10, k=6, w_type = "Majorana", max_weight=1)

    ket = Ket(N,0)
    o = Pauli(N, Z=[1])

    i = 4
    α = i * π /32
    #Create generators and parameters for the model
    generators, parameters = heisenberg_1D(o, Jx = 0.8, Jy = 0.9,Jz = 0.9, k=k)

    
    #Call to bfs bfs_evolution_test based on weight
    
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
    output_dir = "HEISENBERG_series_results",
    txt_filename = "data_1d_bfs_heisenberg_series.txt"
)

    # Make directory if it doesn't exist
    isdir(output_dir) || mkpath(output_dir)

    txt_file_path = joinpath(output_dir, txt_filename)
    open(txt_file_path, "w") do io
        for N in Ns
            for k in ks
                println("Running N = $N, k = $k")
                max_weights = 1:1:N
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