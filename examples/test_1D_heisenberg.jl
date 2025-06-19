# 
# - - - 1D Heisenberg Spin lattice model - - - 
#
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

function run(; N=10, k=6, w_type = "Majorana", max_weight=1)

    ket = Ket(N,0)
    o = Pauli(N, Z=[1])

    i = 4
    α = i * π /32
    #Create generators and parameters for the model
    generators, parameters = heisenberg_1D(o, Jx = 0.8, Jy = 0.9,Jz = 0.9, k=k)
    #generators, parameters = heisenberg_1D(o, Jx = 0.0, Jy = 0.0,Jz = 1.0, k=k)
    
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

    filename="1D_Heisenberg_abs_error_vs_weight_$w_type-N=$N-k=$k.pdf"
    savefig(plt, filename)
    println("Plot saved as $filename")
end
    
plot_abs_error_vs_weight_pdf(N=6, k=10, w_type="Majorana", max_weights=1:1:10)
