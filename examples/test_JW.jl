using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators

"""
 The following function performs the Jordan-Wirgner mapping for fermionic 
    bilinear terms 
    N - Total number of fermionic modes
    a,b, - indices of the modes to be mapped
"""
function JWmapping(a::Int, b::Int, N::Int)
    # Compute C^dagger_i term
    ax_term = Pauli(2^(a-1)-1, 2^(a-1), N)
    ay_term = Pauli(2^(a)-1, 2^(a-1), N)
    c_dagg_a = 0.5 * (ax_term - ay_term)
#    println("c_dagg_a = 0.5 * ", string(ax_term), " - ", string(ay_term))

    # Compute C_j term
    bx_term = Pauli(2^(b-1)-1, 2^(b-1), N)
    by_term = Pauli(2^(b)-1, 2^(b-1), N)
    
    c_b = 0.5 * (bx_term + by_term)
#    println("c_b = 0.5 * ", string(bx_term), " + ", string(by_term))

    # Build C^dagger_i*C_j
    term =  c_dagg_a*c_b

    return term
end

function creation_FOp_JWmapping(a::Int, N::Int)
    # Compute C^dagger_i term
    ax_term = Pauli(2^(a-1)-1, 2^(a-1), N)
    ay_term = Pauli(2^(a)-1, 2^(a-1), N)
    c_dagg_a = 0.5 * (ax_term - ay_term)
   # println("c_dagg_a = 0.5 * ", string(ax_term), " - ", string(ay_term))

    return c_dagg_a
end

function annihilation_FOp_JWmapping(b::Int, N::Int)
    # Compute C_j term
    bx_term = Pauli(2^(b-1)-1, 2^(b-1), N)
    by_term = Pauli(2^(b)-1, 2^(b-1), N)
    c_b = 0.5 * (bx_term + by_term)
  #  println("c_b = 0.5 * ", string(bx_term), " + ", string(by_term))

    return c_b
end

println("#################################")
c_dag_1 = creation_FOp_JWmapping(1, 4)
println("Creation c^dag_1:")
display(c_dag_1)
#for i in c_dag_1
#3    println(string(i))
#end

c_1 = annihilation_FOp_JWmapping(4, 4)
println("Annihilation c_4:")
display(c_1)
#for i in c_dag_1
#    println(string(i))
#end

println("c_dagg_1* c_4:")
C_11 = JWmapping(1, 4, 4)
display(C_11)

#######################3
function jw_transform(o::Pauli{N}, site) where N
    z_string = [i for i in 1:site-1]
    # p = PauliSum(N)
    p = Pauli(N, Z = z_string, X = [site]) + im * Pauli(N, Z=z_string, Y=[site])
    return 0.5*p
end
println("#############################")
N =4
o = o = Pauli(N, Z=[1])
C_1 = jw_transform(o, 1)
C_4 = jw_transform(o, 4)
println("Alternative version:")
println("Creation c^dag_1:")
display(C_1')
println("Annihilation c_4:")
display(C_4)
println("c_dagg_1* c_4:")
display(C_1'*C_4)