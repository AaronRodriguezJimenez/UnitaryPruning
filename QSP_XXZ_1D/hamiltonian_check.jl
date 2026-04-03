using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf

#
#- - - Hamiltonian - - -
#
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
    coeff_clip!(H)
    return H
end

# Define model parameters
Jx = 1.0
Jy = 1.0
Jz = 1.0
gx = 0.0
gy = 0.0
gz = 0.0
g = 0.00#-0.01
N = 10 #Total number of qubits
H = heisenberg_1D(N, Jx, Jy, Jz)
@printf("1D-Heisenberg Hamiltonian (J= %.2f, Jz= %.2f): \n", Jx, Jz)
display(H)

Hmat = Matrix(H)
#println("Lowest Eigenvalues:")
#for (idx, eig) in enumerate(eigvals(Hmat))#[1:5])
#    @printf("  Eig %1i: %.6f \n", idx, eig)
#end

# 1) diagonalize H
E = eigvals(Hmat)
E = sort(real(E))   # sorted ascending

# 2) print first few eigenvalues and consecutive gaps
Kshow = min(20, length(E))
@printf("\nFirst %d eigenvalues:\n", Kshow)
for i in 1:Kshow
    @printf("  %2d  E = %12.8f\n", i-1, E[i])
end

@printf("\nConsecutive gaps (first %d):\n", Kshow-1)
for i in 1:Kshow-1
    gap = E[i+1] - E[i]
    @printf("  gap %2d->%2d = %12.8f\n", i-1, i, gap)
end

# 3) build list of positive pairwise differences among the lowest Mlevels
Mlevels = min(200, length(E))   # levels to include
Δs = Float64[]
for m in 1:Mlevels
    for n in 1:Mlevels
        d = E[n] - E[m]
        if d > 1e-12
            push!(Δs, d)
        end
    end
end
Δs = unique(sort(Δs))
@printf("\nSmallest 20 positive pairwise differences among first %d levels:\n", Mlevels)
for i in 1:min(20,length(Δs))
    @printf("  Δ%2d = %12.8f\n", i-1, Δs[i])
end