using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf

function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
end

"""
USING MATRIX ALGEBRA
 Here we compute the quantum signal as: C(t) = <psi_0 | O(0) O(t) | psi_0>
 where O(0) is an initial operator and O(t) the operator evolved at time t.
 psi_0 is an initial state. The evolution is performed for the 1D transverse field Ising model
 H = -J * (Sum_{<i,j>}(X_i X_j + Y_i Y_j) -Jz * (Sum_{<i,j>}(Z_i Z_j) + x * Sum_j(X_j)) y * Sum_j(Y_j)) + z * Sum_j(Z_j)).
"""
function heisenberg_1D(N, Jx, Jy, Jz; x=0, y=0, z=0)
    H = PauliSum(N, Float64)
    for i in 0:N-1
        H += -Jx * Pauli(N, X=[i+1,(i+1)%(N)+1])
        H += -Jy * Pauli(N, Y=[i+1,(i+1)%(N)+1])
        H += -Jz * Pauli(N, Z=[i+1,(i+1)%(N)+1])
    end 
    for i in 1:N
        H += x * Pauli(N, X=[i])
        H += y * Pauli(N, Y=[i])
        H += z * Pauli(N, Z=[i])
    end 
    coeff_clip!(H, thresh=1e-16)
    return H
end

#- - - - - - - - - - - - - - -

# Step 1: Build the ingredients
# Define Hamiltonian parameters
J = 1.0
g = 0.00#-0.01
N = 10 #Total number of qubits
H = heisenberg_1D(N, J, J, J) #Isotropic Heisenberg model 
@printf("1D-XXZ Hamiltonian (J= %.2f): \n", J)
display(H)

Hmat = Matrix(H)
println("Lowest Eigenvalues:")
for (idx, eig) in enumerate(eigvals(Hmat))#[1:5])
    @printf("  Eig %1i: %.6f \n", idx, eig)
end

