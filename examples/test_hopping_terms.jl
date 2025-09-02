using LinearAlgebra, Test

"""
 Unit tests for Jordan-Wigner mapping and fermionic operators
"""

# Build exact fermionic operators as matrices
function fermionic_creation(N, i)
    # Build 2^N x 2^N matrix
    dim = 2^N
    mat = zeros(ComplexF64, dim, dim)
    for state in 0:dim-1
        # check if site i is occupied
        if iszero((state >> (i-1)) & 0x1)
            new_state = state | (1 << (i-1))
            # fermionic sign = number of 1's below i
            sign = (-1)^(count_ones(state & ((1 << (i-1))-1)))
            mat[new_state+1, state+1] = sign
        end
    end
    return mat
end

function fermionic_annihilation(N, i)
    # c_i = (c_i^\dagger)†
    return fermionic_creation(N, i)'
end

function fermionic_bilinear(N, i, j)
    return fermionic_creation(N, i) * fermionic_annihilation(N, j)
end

@testset "JWmapping tests" begin
    for N in 2:3
        for i in 1:N, j in 1:N
            # JW result
            jw_op = UnitaryPruning.JWmapping(Pauli(N), i=i, j=j)
            jw_mat = Matrix(jw_op)

            # exact fermionic bilinear
            f_op = fermionic_bilinear(N, i, j)

            @test isapprox(jw_mat, f_op; atol=1e-10)

            # Vacuum test
            vac = zeros(ComplexF64, 2^N); vac[1] = 1
            res = jw_mat * vac
            @test norm(res) < 1e-10

            # Special case: number operator (i=j)
            n_op = fermionic_creation(N, i) * fermionic_annihilation(N, i)
            @test isapprox(Matrix(UnitaryPruning.JWmapping(Pauli(N), i=i, j=i)), n_op; atol=1e-10)
        end
    end

    # Hermiticity test for hopping
    N = 8
    hop = UnitaryPruning.JWmapping(Pauli(N), i=1, j=2) + UnitaryPruning.JWmapping(Pauli(N), i=2, j=1)
    @test ishermitian(Matrix(hop))
end
