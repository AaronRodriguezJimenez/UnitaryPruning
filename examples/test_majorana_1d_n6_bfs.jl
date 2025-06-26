# Example usage PauliOperators
# Based on examples > plot_1d_n6_bfs.jl
using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators


function clip_majorana_weight!(p::Dict{PauliBasis{N}, T}; max_length=4) where {N, T<:Number}
    filter!((k,v) -> PauliOperators.simple_majorana_weight(v) < max_length, p)
end

function clip_pauli_weight!(p::Dict{PauliBasis{N}, T}; max_length=4) where {N, T<:Number}
    filter!((k,v) -> PauliOperators.pauli_weight(v) < max_length, p)
end


"""
 This function removes small terms (based on their magnitude) from a PauliSum, which is represented as a dictionary
"""
function clip_thresh!(p::Dict{PauliBasis{N}, T}; thresh=1e-6) where {N, T<:Number}
    for k in collect(keys(p))
        if abs(p[k]) < thresh
            delete!(p, k)
        end
    end
    return p
end

"""
 This function removes terms larger than a given weight from a PauliSum.
"""
function clip_weight!(p::Dict{PauliBasis{N}, T}; weight=4) where {N, T<:Number}
    for k in collect(keys(p))
        println("Test clip :", PauliOperators.pauli_to_majorana_occupation(k)[1])
        if PauliOperators.pauli_to_majorana_occupation(k)[1] > weight
            delete!(p, k)
        end
    end
    return p
end

Pbasis = PauliBasis("XYZ")
print(Pbasis)
w = PauliOperators.simple_majorana_weight(Pbasis)
println("Simple Majorana weight is: ", w)

w_two = PauliOperators.pauli_to_majorana_occupation(Pbasis)
println("Majorna weight ver 2 is: ", w_two)

println("Testing commute function")
println(PauliOperators.commute(Pbasis, Pbasis))


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
            pi = Pauli{N}(-pi.s, pi.z, pi.x)
            push!(generators, pi)
            push!(parameters, α)
        end
    end

    return generators, parameters
end

"""
 Just for testing purposes
   bfs_evolution_test(generators::Vector{Pauli{N}}, angles, o::Pauli{N}, ket ; thres=1e-3) where {N}

"""
function bfs_evolution_test(generators::Vector{Pauli{N}}, angles, o::PauliSum{N}, ket ; max_m_weight=max_m_weight) where {N}

    #
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    
    vcos = cos.(angles)
    vsin = sin.(angles)

    # collect our results here...
    expval = zero(ComplexF64)


    o_transformed = deepcopy(o)
  
    n_ops = zeros(Int,nt)
    majo_thresh = 2
    for t in 1:nt

        g = generators[t]
        
        sin_branch = PauliSum(N)
        #println("Change ... Aaron was here")
        #println("o_transformed ", o_transformed)

        for (oi, coeff) in o_transformed#.ops
            #println("coeff ", coeff, " oi: ", oi, "generator ", PauliBasis(g))
            #- - - INSERT HERE THE MAJORANA-WEIGHT CRITERIA? - - -
            #println("Conversion to Majorana:")
            #print(PauliOperators.string(oi))
            #println(PauliOperators.pauli_to_majorana_occupation(oi))
            
            # majo_weight = PauliOperators.pauli_to_majorana_occupation(oi)[1]
            # majo_weight <= majo_thresh || continue
            #abs(coeff) > thresh || continue

            if PauliOperators.commute(oi, PauliBasis(g)) == false
                #println("oi and g commute")
            #if commute(oi, g.pauli) == false #original


                # cos branch
                o_transformed[oi] = coeff * vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's
                if PauliOperators.simple_majorana_weight(oj) > max_m_weight
                    println("m weight larger than max")
                    continue
                end
                sum!(sin_branch, oj * vsin[t] * coeff * 1im)

            end
        end
        sum!(o_transformed, sin_branch) 
        #clip_thresh!(o_transformed, thresh=thresh)
        clip_weight!(o_transformed, weight=majo_thresh)
 #       clip_majorana_weight!(o_transformed, weight=majo_thresh)
        n_ops[t] = length(o_transformed)
    end

    for (oi,coeff) in o_transformed#.ops
        expval += coeff*expectation_value(oi, ket)
    end
   
    return expval, n_ops
end

#
#   in the experiment, the circuit is 
#
#   exp(i θ/2 (-X)) exp(i π/4 ZZ)
#

function run(; N=6, k=10, max_weight=4)
   
    ket = Ket(N, 0) 
    o = Pauli(N, Z=[1])

    angles = [] 
    e = [] 
    
    # for i in [(i-1)*2 for i in 1:9]
    for i in 0:16
        α = i * π / 32 
        generators, parameters = get_unitary_sequence_1D_test(o, α=α, k=k)
        
        #for (j,g) in enumerate(generators)
        #    println("Generator :", j, " ", PauliBasis(g))
        #end

        ei , nops = bfs_evolution_test(generators, parameters, PauliSum(o), ket, max_m_weight=max_weight)
      
        push!(e, ei)
        push!(angles, α)
        @printf(" α: %6.4f e: %12.8f+%12.8fi nops: %i\n", α, real(ei), imag(ei), maximum(nops))
        
    end
    
    plot(angles, real(e), marker = :circle)
    xlabel!("Angle")
    ylabel!("expectation value")
#    title!{X_{13,29,31}, Y_{9,30}, Z_{8,12,17,28,32}}
    #savefig("plot_1d_n6_bfs.pdf")
    savefig("plot_1d_n6_bfs_majorana.pdf")
    return e
end

@time v,e = run(k=6, N=6, max_weight=4);