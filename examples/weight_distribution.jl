using Distributed
using UnitaryPruning
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators
using Plots

"""
  Here, we explore the distribution of Pauli weight for different time evolutions.
  This initial tests is performed using the linear XXZ model.
"""

"""
 1D linear chain version of the Heisenberg model
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

function clip_coeff!(p::Dict{PauliBasis{N}, T}; thresh=1e-3) where {N, T}
    filter!(p-> abs(p.second) ≥ thresh, p)
end

"""
    Instead of performing the expectation value estimation, this fucntion runs a propagation at a set of parameters
    and returns the weights and coefficients of the pauli strings generated during the propagation
"""
function bfs_evolution_distribution(generators::Union{Vector{Pauli{N}},Vector{PauliBasis{N}}}, angles, o::PauliSum{N}; thresh=1e-3) where {N}
#
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    
    vcos = cos.(angles)
    vsin = sin.(angles)

    # collect our results here...
    weights = Int[]
    coeffs = ComplexF64[]
    n_ops = zeros(Int, nt)
    tot_nops = 0

    o_transformed = o #deepcopy(o)    #we are Ok if this mutates

    for t in 1:nt
        #println("Evaluating angle $t of $nt")
        g = generators[t]
        pb = PauliBasis(g)
        sin_branch = PauliSum(N)

        for (oi, coeff) in o_transformed

            abs(coeff) > thresh || continue

            if !PauliOperators.commute(oi, pb)
            
                # cos branch
                o_transformed[oi] = coeff .* vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's

                sum!(sin_branch, oj * vsin[t] * coeff * 1im)
            end
        end
        sum!(o_transformed, sin_branch)
        clip_coeff!(o_transformed, thresh=thresh)

        #Check if sin_branch is empty
        #!isempty(sin_branch) && sum!(o_transformed, sin_branch)

        # We compute the weights at each transformation, this may account for
        # all the possible operators generated in the evolution:
        len = length(o_transformed)        
        n_ops[t] = len
        tot_nops += len
        
        for (oi, coeff) in o_transformed
            #display(oi)
            #println(coeff)
            push!(weights, PauliOperators.pauli_weight(oi))
            push!(coeffs, coeff)

        end
    end
    return weights, coeffs, tot_nops
end


function run(N=10, k=6, thresh=1e-3)

    o = Pauli(N, Z=[1])
    Jx = 1.0
    Jy = 1.0
    Jz = 0.5

    #Create generators and parameters for the model
    generators, parameters = heisenberg_1D(o, Jx = Jx, Jy = Jy, Jz = Jz, k=k)
   
    #Call to bfs_evolution
    weights, coeffs, tot_nops = bfs_evolution_distribution(generators, parameters, PauliSum(o), thresh=thresh)

    return weights, coeffs, tot_nops
end


#Ns = [8,16,32,64,128]
#ks = [1,2,3,4,5]
Ns = [16]
ks = [1]
for N in Ns
    for k in ks
        coeff_threshold = 0.0001
        w, c, n_ops = run(N, k, coeff_threshold)
        
        w_int = Int.(w)
        μ = mean(w_int)
        σ = std(w_int)
        println("Result for N=$N, k=$k")
        println("Total -ops > $coeff_threshold : $n_ops")
        println("Mean weight: $μ")
        println("SDTV       : $σ")
        println("- - - - - - - -")

        # Compute bin edges to center bars on integers
        w_int = Int.(w)
        min_w = minimum(w_int)
        max_w = maximum(w_int)
        edges = (min_w - 0.5):(max_w + 0.5)
        xtick_vals = min_w:max_w

        p = histogram(w_int;
        normalize = true,
        bins = edges,
        xticks = (xtick_vals, string.(xtick_vals)),
        xlabel = "Pauli Weight",
        ylabel = "Normalized Frequency",
        title = "Weight Distribution (k = $k, thresh=$coeff_threshold)",
        bar_width = 0.7,
        grid = :y,
        label = "",
        guidefontsize = 14,
        tickfontsize = 10,
        legendfontsize = 10,
        fillcolor = :gray,
        )
        
        vline!([μ], label = "Mean", color = :black, linestyle = :dot, linewidth = 3)
        vline!(p, [μ - σ, μ + σ]; label = "±σ", linestyle = :dot, color = :red, linewidth = 2)
        savefig(p, "pauli_weights_histogram_N=$N-k=$k.pdf")

    end
end
        
#=N = 128
k = 5
coeff_threshold = 0.001
w, c, n_ops = run(N, k, coeff_threshold)

#println("- - - Weights - - -")
#println(w)
#println("- - - Coeffs - - -")
#println(c)

"""
  Now that we have the weight for all the operators generated during the propagation
  we can generate an histogram to record their apparition.
"""
# Compute bin edges to center bars on integers
w_int = Int.(w)
min_w = minimum(w_int)
max_w = maximum(w_int)
edges = (min_w - 0.5):(max_w + 0.5)
xtick_vals = min_w:max_w

# Compute mean and standard deviation
μ = mean(w_int)
σ = std(w_int)=#

#=p = histogram(w_int;
    normalize = true,
    bins = edges,
    xticks = (xtick_vals, string.(xtick_vals)),
    xlabel = "Pauli Weight",
    ylabel = "Normalized Frequency",
    title = "Weight Distribution (k = $k, thresh=$coeff_threshold)",
    bar_width = 0.7,
    grid = :y,
    label = "",
    guidefontsize = 14,
    tickfontsize = 10,
    legendfontsize = 10,
    fillcolor = :gray,
)=#

#Overlay mean line
#vline!([μ], label = "Mean", color = :black, linestyle = :dot, linewidth = 3)
#Overlay std dev lines
#vline!(p, [μ - σ, μ + σ]; label = "±σ", linestyle = :dot, color = :red, linewidth = 2)

#savefig(p, "pauli_weights_histogram_N=$N-k=$k.pdf")