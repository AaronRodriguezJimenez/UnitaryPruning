using UnitaryPruning
using KrylovKit
using LinearAlgebra
using PauliOperators
using BenchmarkTools
using Random

#
# Testing exact exponentiation time evolution
#
N = 2
generators = [Pauli(N, Z=[1], X=[2])]#, Pauli(N, X=[1])]
parameters = [3.14/2]#, 0.6]
o = Pauli(N, Z=[2])
ket = Ket(N,2)
bra = Bra(N,1)

mem_U = @allocated U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
o_mat = Matrix(o)
m = diag(U'*o_mat*U)

println("- - - Allocated Memory for U- - - -")
println("$(mem_U*1e-06) MB")
println("- - - U[1] Matrix entry - - - -")
println(U[1])
println("- - - o  - - - -")
display(o)
println("- - - m  = diag(U'oU) - - - -")
println(m)
println("- - - m[1] - - - -")
println(m[1])

##

"""
  matvec fucntion return the effect of Operator o applied to the vector V
  which must correspond to some compatible ket    
"""
function matvec(o::Pauli{N}, coeff::Number, V::Vector) where N
    T = promote_type(eltype(V), typeof(coeff))
    σ = zeros(T, size(V))

    for i in 0:2^N - 1
        ketj = o * Ket(N, i)  # returns (coeff, Ket)
        j = ketj[2].v
        σ[j + 1] += coeff * V[i + 1] * ketj[1]
    end

    return σ
end

# Usage
#println("Ket: ", ket, ' ', typeof(ket))
vector_ket = Vector(ket)
println("Vector ket: $ket -> ", vector_ket)
phi = 1.00
operation = matvec(o, phi, Vector(ket))
println("matvect operation :  " ,operation, " ", typeof(operation))

# Next we focus on compute the previous element m[1] in the Shcrodinger picture..
function compute_schrodinger(generators, parameters, ref_ket)
    nt = length(generators)
    length(parameters) == nt || throw(DimensionMismatch)
    U_psi = 0
    for t in 1:nt
        α = parameters[t]
        #U |psi_0> : cos(α/2) * ψ - i*sin(α/2)*Pψ
        U_psi = cos(α/2) .* matvec(generators[t], α, ref_ket) .- 1im*sin(α/2) .* matvec(generators[t], α, ref_ket)
        ref_ket = matvec(generators[t], α, ref_ket)
    end
    return U_psi / norm(U_psi)
end

U_psi = compute_schrodinger(generators, parameters, vector_ket)
println("res :", U_psi)

expval = U_psi' * matvec(o, 1.00, U_psi)
println("Expval Schr :", expval)

expval_Heis = vector_ket' * (U' * o_mat * U) * vector_ket
println("Expval Heis :", expval_Heis)