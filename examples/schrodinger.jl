using UnitaryPruning
using LinearAlgebra
using PauliOperators
using BenchmarkTools
using ExponentialUtilities  
"""
  matvec fucntion return the effect of Operator o applied to the vector V
  which must correspond to some ket    
"""
function apply_pauli_index_phase(p::Pauli{N}, i::Int) where N
    coeff, ketj = p * Ket(N, i)  
    
    return ketj.v, coeff
end

function matvec(o::Pauli{N}, coeff::Number, V::Vector) where N
    σ = zeros(promote_type(typeof(coeff), eltype(V), ComplexF64), length(V))

    for i in 0:2^N - 1
        j, phase = apply_pauli_index_phase(o, i)  # returns target index and phase
        
        σ[j + 1] += coeff * V[i + 1] * phase
    end

    return σ
end

# Next we focus on compute the previous element m[1] in the Shcrodinger picture..
function compute_schrodinger_evol(generators, parameters, ref_ket)
    nt = length(generators)
    length(parameters) == nt || throw(DimensionMismatch)

    U_psi = copy(ref_ket)

    for t in 1:nt
        α = parameters[t]
        Pψ = matvec(generators[t], 1.0, U_psi)
        U_psi = cos(α/2) .* U_psi - 1im * sin(α/2) .* Pψ
        U_psi /= norm(U_psi)  # Optional: normalize at each step
    end

    return U_psi
end


## Testing Function matvec
p = Pauli("YI")  # or any multi-qubit Pauli
v = ComplexF64[1.0, 0.0, 0.0, 0.0]  # example: |00⟩

# Direct matrix multiplication
Mv = Matrix(p) * v

# Implementation
mv = matvec(p, 1.0, v)

# Compare
println("M*v vs m*v - Are they equal? ", isapprox(Mv, mv))

θ = π/3
U = exp(-1im * θ/2 * Matrix(p))  # exact unitary
ψ1 = U * v                       # reference output

# Approximate it manually
ψ2 = cos(θ/2) * v - 1im * sin(θ/2) * matvec(p, 1.0, v)

println("U*v vs cos(θ/2) * v - 1im * sin(θ/2) * matvec(p, 1.0, v) - Are they equal? ", isapprox(ψ1, ψ2))
println(" ")

# # # # # Test Time evolution subroutines # # # # # #
N = 5
generators = [Pauli(N, Z=[1], X=[2]), Pauli(N, Z=[1,2]), Pauli(N, Z=[2])]
parameters = [3.14/2, 0.6, 0.3]
o = Pauli(N, Z=[2])
ket = Ket(N,2)
bra = Bra(N,1)

vector_ket = Vector(ket)
println("Vector ket: $ket -> ", vector_ket)

U_psi = compute_schrodinger_evol(generators, parameters, vector_ket)
println("res :", U_psi)

expval = U_psi' * matvec(o, 1.00, U_psi)
println("Expval Schr :", expval)


#mem_U = @allocated U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
#o_mat = Matrix(o)
#m = diag(U'*o_mat*U)
#expval_Heis = vector_ket' * (U' * o_mat * U) * vector_ket
#println("Expval Heis :", expval_Heis)


# Expval - Approximate it manually
#θ = parameters[1]
#g = generators[1]
#ψ2 = cos(θ/2) .* vector_ket - 1im * sin(θ/2) .* matvec(g, 1.0, vector_ket)
#ψ2 = ψ2/norm(ψ2)
#expval = ψ2' * matvec(o, 1.0, ψ2)
#println(expval)