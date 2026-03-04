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
    coeff_clip!(H)
    return H
end

# return a PauliOperators Ket equivalent to a given bitstring
function string_to_ket(bits::String)
    b = collect(bits)
    v = parse.(Int128, b)
    N = length(v)
    out = 0
    count = 0

    for bit in v
        if bit%2 == 1
            out += 2^count
        end
        count +=1
    end
    ket = Ket{N}(out)
    return ket, out
end
#- - - - - - - - - - - - - - -

# Step 1: Build the ingredients
# Define Hamiltonian parameters
J = 1.0
Jz = 2.0
g = 0.00#-0.01
N = 10 #Total number of qubits
H = heisenberg_1D(N, J, J, Jz)
@printf("1D-XXZ Hamiltonian (J= %.2f, Jz= %.2f): \n", J, Jz)
display(H)

Hmat = Matrix(H)
println("Lowest Eigenvalues:")
for (idx, eig) in enumerate(eigvals(Hmat))#[1:5])
    @printf("  Eig %1i: %.6f \n", idx, eig)
end

# Define initial ket and Initial operator to be evolved under the circuit
ket = Ket{N}(1)
#ket = Ket{N}(12)
#ket, _ = string_to_ket("1000")
#o = Pauli(N) # The identity
o = Pauli(N, X=[1])
#o = Pauli(N, X=[3,4])
#o += Pauli(N,X=[1,2]) #Amplitude modulation
#o += Pauli(N,X=[1,2,3,4]) #mixed signal
#o += Pauli(N,X=[3])  #mixed signal
#o += Pauli(N,X=[2,3]) #Imaginary suppression

Omat = Matrix(o)
V0 = Vector(ket)

E = eigen(Hmat)
λ = E.values      # eigenvalues
V = E.vectors     # eigenvectors (columns)
display(ket)
display(V)

# Compute O matrix elements:
# O_{a0} = <v_a | O | v_0>
Oa0 = V' * (Omat * V0)
println("O_{a0} = <v_a | O | v_0>")
display(Oa0)

# Time evolution unitaries:
function Ut(Hmat, dt)
    return exp(-im * dt * Hmat)
end

# Time grid
# Define time evolution parameters
# Circuit divided in k layers
# Thus total time (t) is divided in dt = t/k time intervals
k = 100
t = 2.5
dt = t/k
nsamp = Int(k) + 1
tgrid = collect(range(0.0, stop=k*dt, length=nsamp))

# Step 2: Compute C(t)
W = deepcopy(Omat)
Ct = Vector{ComplexF64}([])
for time in tgrid
    U = Ut(Hmat, time)
    Udag = U'
    UdagWU = Udag * W * U
    WWt = W * UdagWU
    res = V0' * WWt * V0
   # res = res * (1/norm(res))
    push!(Ct, res)
end

#Step 3: Plot C(t)
println("- - - C(t) - - - ")
#display(Ct)

rRES = real(Ct)
iRES = imag(Ct)
plt = plot(tgrid, rRES, lw=2, seriestype=:scatter,
          label="Re(C(t)")
plt = plot!(tgrid, iRES, lw=2, seriestype=:scatter,
          label="Im(C(t)")

xlabel!(plt, "Time"); ylabel!(plt, "< O(0)O(t) >")
title!(plt, "N=$N, J=$J, g=$g,dt=$dt")
savefig(plt, "QSP_XXZ_1D_exact.pdf")

println("- - - Sanity Check: |C(t)|^2 - - - ")
@printf("idx    dt     Re(C(t))    Im(C(t))   |C(t)|^2   |C(t)|\n")
for (i,interval) in enumerate(tgrid)
    normop2 = rRES[i]^2 + iRES[i]^2
    normop = sqrt(normop2)
    @printf("%.2s    %.4f    %.6f   %.6f   %.6f  %.6f\n", i, interval, rRES[i], iRES[i], normop2, normop)
end


#- - - CHECK RESULT AT t = 0
time = 0
U = Ut(Hmat, time)
Udag = U'
W = deepcopy(Omat)
Wt = Udag * W * U
WWt = W' * Wt
res = V0' * WWt * V0
println("Result at t = 0: ")
println("V0' * WWt * V0 : ", res )
println("V0' * Wt * V0: ", V0' * Wt * V0)
println("V0' * W * V0: ", V0' * W * V0)
#display(Wt)

#- - - - - PHASE CORRECTION (ONLY WHEN REF STATE IS ALSO AN EIGENSTATE) - - - - -
# SIGNAL PROCESSING multiply by exp(iE_kt) to correct signal
Ek = 0.0#-8 #Eigenvalue associated to |0000> state
signal = rRES .+ 1im * iRES;
phase = exp.(1im *Ek .* tgrid); #-1 is the eigenvalue associated with the refernce eigenvector (E_k)

#corrected signal F(t) = exp(-iE_0t)*C(t)
F = phase .* signal

# Print C(t) results
plt2 = plot(tgrid, real(F), lw=2, seriestype=:scatter,
          label="Re(F(t)")
plt2 = plot!(tgrid, imag(F), lw=2, seriestype=:scatter,
          label="Im(F(t)")

xlabel!(plt2, "Time"); ylabel!(plt2, "exp(-iE_kt) * < O(0)O(t) >")
title!(plt2, "N=$N, J=$J, Jz=$Jz,dt=$dt")
savefig(plt2, "QSP_XXZ_1D_corrct_exact.pdf")

println("- - - Compare signals  - - - ")
@printf("dt   Re(C(t))    Im(C(t))   Re(F(t))    Im(F(t))\n")
for (i,interval) in enumerate(tgrid)
    normop2 = rRES[i]^2 + iRES[i]^2
    normop = sqrt(normop2)
    @printf("%.4f     %.6f    %.6f    %.6f   %.6f\n", interval, rRES[i], iRES[i], real(F[i]), imag(F[i]))
end

plt3 = plot(tgrid, rRES, lw=2, seriestype=:scatter,
          label="Re(C(t)")
plt3 = plot!(tgrid, iRES, lw=2, seriestype=:scatter,
          label="Im(C(t)")
plt3 = plot!(tgrid, real(F), lw=2, seriestype=:scatter,
          label="Re(F(t)")
plt3 = plot!(tgrid, imag(F), lw=2, seriestype=:scatter,
          label="Im(F(t)")

xlabel!(plt3, "Time"); ylabel!(plt3, "exp(-iE_kt) * < O(0)O(t) >")
title!(plt3, "Signal Comparison N=$N, J=$J, g=$g,dt=$dt")
savefig(plt3, "QSP_XXZ_1D_comparison_exact.pdf")

## Compare signal with "theoretical one"
#
# According to the hypotheses, the correct signal must be
#f(x) = (1/4)*cos(4x) + (1/2)*cos(-8x) + (1/4)*cos(12x)
#
# Evaluate on the grid
#y = f.(tgrid)
# Plot
#plt4 = plot(tgrid, rRES, lw=2, seriestype=:scatter,
#          label="Re(C(t)")
#plt4 = plot!(tgrid, y, lw=2, xlabel="t", ylabel="C(t)",  
#             title="Signal vs theoretical f(x)", label="f(t)",
#             legend=true)
#
#savefig(plt4, "QSP_XXZ_signal_vs_theory.pdf")