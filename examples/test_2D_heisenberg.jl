# 
# - - - 2D Heisenberg Spin lattice model - - - 
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
 2D Heisenberg model with periodic boundary conditions.
 Similar as the previous function, however, here we wrap around using 
 modulo arithmetic
 x_right wraps L -> 1
 y_down wraps L -> 1
 This means: site (L,y) connects to (1,y) & site(x,L) connects to (x,1.
"""
function heisenberg_2D(o::Pauli{N}; Jx, Jy, Jz, k) where N
    L = Int(sqrt(N))
    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()
    index(i, j) = (mod1(i, L) - 1) * L + mod1(j, L)
    for ki in 1:k
        for i in 1:L, j in 1:L
            a = index(i, j)
            # Right neighbor (periodic in j)
            b = index(i, j + 1)
            push!(generators, Pauli(N, X=[a, b])); push!(parameters, Jx)
            push!(generators, Pauli(N, Y=[a, b])); push!(parameters, Jy)
            push!(generators, Pauli(N, Z=[a, b])); push!(parameters, Jz)
            # Down neighbor (periodic in i)
            b = index(i + 1, j)
            push!(generators, Pauli(N, X=[a, b])); push!(parameters, Jx)
            push!(generators, Pauli(N, Y=[a, b])); push!(parameters, Jy)
            push!(generators, Pauli(N, Z=[a, b])); push!(parameters, Jz)
        end
    end
    return generators, parameters
end

function run(; N=4 ,k=2, w_type = "Majorana", max_weight=1)

    #N = Lx*Ly Square lattice
    ket = Ket(N,0)
    o = Pauli(N, Z=[1])

    i = 4
    α = i * π /32


    #Create generators and parameters for the model
    generators, parameters = heisenberg_2D(o, Jx = 0.8, Jy = 0.9,Jz = 0.9, k=k)
    
    #Call to bfs bfs_evolution_test based on weight
    
    ei, nops = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)
    println("Total Operators: " , length(nops))
    println("Estimated Energy: ", ei)

    # Exact evolution
    U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    o_mat = Matrix(o)
    m = diag(U'*o_mat*U)
    abs_err = abs(real(m[1])- real(ei) )
    println("Estimation :", real(m[1]), " Exact :", real(ei), " Absolute Error: ", abs(real(m[1])- real(ei) ))

    return abs_err
end

function plot_abs_error_vs_weight_pdf(; N=9, k=10, w_type="Pauli", max_weights=1:2:10)
    errors = Float64[]
    weights = Int[]
    
    for mw in max_weights
        println("Evaluating max_weight = $mw")
        err = run(N=N, k=k, w_type = w_type, max_weight=mw)
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

    filename="2D_Heisenberg_abs_error_vs_weight_$w_type-N=$N-k=$k.pdf"
    savefig(plt, filename)
    println("Plot saved as $filename")
end
    
plot_abs_error_vs_weight_pdf(N=9, k=10, w_type="Pauli", max_weights=1:2:20)
