using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators


"""
  Here, we explore the distribution of Pauli weight for different time evolutions.
  This initial tests is performed using the linear XXZ model.
"""

"""
 1D linear chain verison of the Heisenberg model
"""
function heisenberg_1D(o::Pauli{N}; Jx, Jy, Jz, k, t=1.0) where N 
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])
    # Loop over sites
    for ki in 1:k 
        for i in 1:N-1
            push!(generators, Pauli(N, X=[i, i + 1]))
            push!(parameters, t*Jx)
            push!(generators, Pauli(N, Y=[i, i + 1]))
            push!(parameters, t*Jy)
            push!(generators, Pauli(N, Z=[i, i + 1]))
            push!(parameters, t*Jz)
        end
    end

    return generators, parameters
end

"""
    Instead of performing the expectation value estimation, this fucntion runs a propagation at a set of parameters
    and returns the weights and coefficients of the pauli strings generated during the propagation
"""
function bfs_evolution_distribution(generators::Union{Vector{Pauli{N}},Vector{PauliBasis{N}}}, angles, o::PauliSum{N}) where {N}
#
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    
    vcos = cos.(angles)
    vsin = sin.(angles)

    # collect our results here...
    weights = []
    coeffs = []
    n_ops = zeros(Int,nt)
    tot_nops = 0

    o_transformed = deepcopy(o)    

    for t in 1:nt
        println("Evaluating angle $t of $nt")
        g = generators[t]
        
        sin_branch = PauliSum(N)

        for (oi, coeff) in o_transformed#.ops
            #println("coeff ", coeff, " oi: ", oi, "generator ", PauliBasis(g))

            if PauliOperators.commute(oi, PauliBasis(g)) == false
            
                # cos branch
                o_transformed[oi] = coeff * vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's

                sum!(sin_branch, oj * vsin[t] * coeff * 1im)
            end
        end
        sum!(o_transformed, sin_branch) 
        n_ops[t] = length(o_transformed)
        # We compute the weights at each transformation, this may account for
        # all the possible operators generated in the evolution:
        tot_nops += length(o_transformed)
        for (oi, coeff) in o_transformed
            if coeff != 0
               # display(oi)
                w = PauliOperators.pauli_weight(oi)
                push!(weights, w)
                push!(coeffs, coeff)
            end
        end
    end
    return weights, coeffs, tot_nops
end


function run(t,  N=10, k=6)

    o = Pauli(N, Z=[1])
    Jx = 1.0
    Jy = 1.0
    Jz = 0.5

    #Create generators and parameters for the model
    generators, parameters = heisenberg_1D(o, Jx = Jx, Jy = Jy, Jz = Jz, k=k, t=t)
    #generators = [Pauli(N, X=[1,2]), Pauli(N, Y=[1,2]), Pauli(N, Z=[1,2])]
    #parameters = [t*Jx, t*Jy, t*Jz]
   
    #Call to bfs_evolution
    weights, coeffs, tot_nops = bfs_evolution_distribution(generators, parameters, PauliSum(o))
    coeffs = coeffs ./ t

    return weights, coeffs, tot_nops
end

t = pi
N = 14
k = 2

w, c, n_ops = run(t, N, k)

#println("- - - Weights - - -")
#println(w)
#println("- - - Coeffs - - -")
#println(c)
#println("- - - Total operators - - -")
#println(n_ops)


"""
  Now that we have the weight for all the operators generated during the propagation
  we can generate an histogram to record their apparition.
"""
# Compute bin edges to center bars on integers
w_int = Int.(w)  # Convert Any[...] to concrete Int[]
min_w = minimum(w_int)
max_w = maximum(w_int)
edges = (min_w - 0.5):(max_w + 0.5)

# Set xticks at integer centers
xtick_vals = min_w:max_w

# Plot
p = histogram(w_int;
    normalize=true,
    bins = edges,
    xticks = (xtick_vals, string.(xtick_vals)),  # Ensure only integers shown
    xlabel = "Pauli Weight",
    ylabel = "Frequency",
    title = "t=$t",
    legend = false,
    bar_width = 0.2
)

savefig(p, "pauli_weights_histogram_N=$N-k=$k.pdf")