using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf
using Random

function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
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

"""
 Here we compute the quantum signal as: C(t) = <psi_0 | O(0) O(t) | psi_0>
 where O(0) is an initial operator and O(t) the operator evolved at time t.
 psi_0 is an initial state. The evolution is performed for the 1D Heisenverg model
 H = -J * (Sum_{<i,j>}(X_i X_j + Y_i Y_j) -Jz * (Sum_{<i,j>}(Z_i Z_j) + gx * Sum_j(X_j)) gy * Sum_j(Y_j)) + gz * Sum_j(Z_j)).

"""
function trott_unitary_sequence_Heisenberg(o::Union{Pauli{N}, PauliSum{N}}; Jx=1.0, 
                                   Jy=1.0, Jz=1.0, gx=0.1, gy=0.1, gz= 0.1, k=10) where N
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])

    # Loop over trotter steps
    for ki in 1:k
        ## XX + YY layer
        for i in 0:N-1
            piX = Pauli(N, X=[i+1,(i+1)%(N)+1])
            push!(generators, piX)
            push!(parameters, -1.0*Jx)

            piY = Pauli(N, Y=[i+1,(i+1)%(N)+1])
            push!(generators, piY)
            push!(parameters, -1.0*Jy) #Positive to account for a change of sign
        end

        ## ZZ layer
        # e^{i π/2 P2} e^{i π/2 P1}|ψ>
        for i in 0:N-1
            piZ = Pauli(N, Z=[i+1,(i+1)%(N)+1])
            push!(generators, piZ)
            push!(parameters, -1.0*Jz)
        end
        
        if gx != 0.0
            ## x layer
            # e^{i αn (-X) / 2}
            for i in 0:N
                Pi = Pauli(N, X=[i])
                push!(generators, Pi)
                push!(parameters, -1.0*gx)
            end
        end

        if gy != 0.0
            ## y layer
            # e^{i αn (-Y) / 2}
            for i in 0:N
                Pi = Pauli(N, Y=[i])
                push!(generators, Pi)
                push!(parameters, -1.0*gy)
            end
        end

        if gz != 0.0
            ## z layer
            # e^{i αn (-X) / 2}
            for i in 0:N
                Pi = Pauli(N, X=[i])
                push!(generators, Pi)
                push!(parameters, -1.0*gz)
            end
        end

    end

    return generators, parameters
end

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

"""
 Evolve function from DBF code
"""
function evolve!(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    sin_branch = PauliSum(N)
    for (p,c) in O
        if PauliOperators.commute(p,G) == false
            # replace sum! with more efficient version
            # sum!(sin_branch, c*_sin*G*p)
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + PauliOperators.coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr 
            O[p] *= _cos
        end
    end
    sum!(O, sin_branch)
    return O 
end

# Helper: extract PauliBasis operators and coefficients from PauliSum
function extract_hamiltonian_coeffs_and_ops(H::PauliSum{N,T}) where {N,T}
    ops = PauliBasis{N}[]
    coeffs = Float64[]
    for (p, c) in H
        push!(ops, p)
        push!(coeffs, float(c))
    end
    return ops, coeffs
end

function weight(p::PauliBasis) 
    return count_ones(p.x | p.z)
end

"""
 Compute the Pauli-weight snapshot
"""
function weight_profile(W::PauliSum{N,T}) where {N,T}
    w = zeros(Float64, N + 1)   # weights 0..N
    total = 0.0
    for (P, c) in W
        k = weight(P)
        a2 = abs2(c)
        w[k + 1] += a2
        total += a2
    end
    if total > 0
        w ./= total #normalize to a probability-like profile.
    end
    return w
end

"""
 Comptue the mean and variance of weight
"""
function weight_stats(w::AbstractVector)
    ks = 0:length(w)-1
    μ = sum(ks .* w)
    σ2 = sum(((ks .- μ).^2) .* w)
    return μ, σ2
end

"""
 DMD structure
"""
struct DMDResult
    A_ls::Matrix{Float64}
    A_tilde::Matrix{ComplexF64}
    modes::Matrix{ComplexF64}
    evals::Vector{ComplexF64}
    amplitudes::Vector{ComplexF64}
    singular_values::Vector{Float64}
    residual_rel::Float64
end

"""
 DMD fit function
"""
function fit_dmd(X::AbstractMatrix{<:Real}; r::Union{Nothing,Int}=nothing, tol=1e-10)
    # X is d x m, columns are snapshots
    X1 = X[:, 1:end-1]
    X2 = X[:, 2:end]

    # Full least-squares map in the snapshot space
    A_ls = Matrix(X2 * pinv(X1))
    residual_rel = norm(X2 - A_ls * X1) / max(norm(X2), eps())

    # Truncated SVD for robust DMD
    F = svd(X1; full=false)
    U = F.U
    s = F.S
    V = F.V

    if r === nothing
        r = count(>(tol * s[1]), s)
        r = max(r, 1)
    else
        r = min(r, length(s))
    end

    Ur = U[:, 1:r]
    sr = s[1:r]
    Vr = V[:, 1:r]

    Sinv = Diagonal(1.0 ./ sr)
    A_tilde = Matrix(Ur' * X2 * Vr * Sinv)

    eig = eigen(A_tilde)
    λ = eig.values
    W = eig.vectors

    # DMD modes
    Φ = Matrix(X2 * Vr * Sinv * W)

    # modal amplitudes from first snapshot
    b = Φ \ complex.(X[:, 1])

    return DMDResult(A_ls, A_tilde, Φ, λ, b, s, residual_rel)
end

function print_dmd_summary(res::DMDResult; dt::Real=1.0, topk::Int=5)
    λ = res.evals
    growth = log.(abs.(λ)) ./ dt
    freq = angle.(λ) ./ dt

    idx = sortperm(abs.(λ), rev=true)
    kmax = min(topk, length(idx))

    println("---- DMD summary ----")
    println("relative LS residual = $(res.residual_rel)")
    println("numerical rank       = $(size(res.A_tilde, 1))")
    println("top singular values   = ", res.singular_values[1:min(end, topk)])

    println("\nDominant modes:")
    for j in 1:kmax
        i = idx[j]
        println("  λ[$i] = $(λ[i])")
        println("      |λ| = $(abs(λ[i]))")
        println("      growth rate = $(growth[i])")
        println("      frequency    = $(freq[i])")
        println("      amplitude    = $(res.amplitudes[i])")
    end
end

"""
 HELPER: Takens delay embedding
 delay parameter here is: q
 X itself is a delay-embedded snapshot matrix.
"""
function delay_embed(snapshots::Vector{<:AbstractVector}, q::Int)
    m = length(snapshots)
    d = length(snapshots[1])
    ncols = m - q + 1
    X = zeros(Float64, d*q, ncols)

    for k in 1:ncols
        for j in 1:q
            X[(j-1)*d+1:j*d, k] .= snapshots[k + j - 1]
        end
    end
    return X
end

"""
 For DMD itself, the most useful statistics are:

  singular values of X: intrinsic dimension
  residual norm: fit quality
  bootstrap spread of dominant eigenvalues: uncertainty
"""
function dmd_bootstrap(snapshots::Vector{<:AbstractVector}; B::Int=100, r::Union{Nothing,Int}=nothing, seed::Int=1)
    rng = MersenneTwister(seed)
    m = length(snapshots)
    d = length(snapshots[1])

    evals = Vector{Vector{ComplexF64}}(undef, B)

    # use contiguous blocks to preserve time ordering
    for b in 1:B
        lo = rand(rng, 1:m-3)
        hi = rand(rng, lo+2:m)
        Xb = hcat(snapshots[lo:hi]...)
        res = fit_dmd(Xb; r=r)
        evals[b] = res.evals
    end

    return evals
end

"""
Let's see if we can track the evolution of O(t) via tacking the weight of the
generated operators in the Pauli basis.
"""
function evolution_op(ket, o::PauliSum{N,T}, H::PauliSum{N,T}, n_intervals, dt;
                      thresh::Float64=1e-3) where {N,T}

    #err = Vector{Float64}([])
    Wt = deepcopy(o)        # evolve W ≡ U*OU
    W  = deepcopy(o)        # initial operator 
    rCtvals = Vector{Float64}([])#(undef, nsamp) # vector for C(t) values storing
    iCtvals = Vector{Float64}([])#(undef, nsamp)
    snapshots = Vector{Vector{Float64}}()

    # t = 0 
    WW = W*W
    expval0 = expectation_value(WW, ket)

    C0real = real(expval0)
    C0imag = imag(expval0)

    push!(rCtvals, C0real)
    push!(iCtvals, C0imag)
    push!(snapshots, weight_profile(W))

    # Extract Pauli strings and coefficients
    generators, angles = extract_hamiltonian_coeffs_and_ops(H)
    #@printf("Hamiltonian has %d terms \n", length(coeffs))

    nt = length(angles)                            
    println("Total Rotations:", nt * n_intervals)
    # Evolve W under Trotterization

    #accumulated_error = 0
    for ki in 1:n_intervals
            # Trotter terms U_1 U_2, ..., U_k
            accumulated_error = 0

            for j in 1:nt
                # Access to the evolution of the operator by H = Sum(theta_i * P_i)
                Pi  = generators[j]
                theta = 2 * dt * angles[j]
                pb = PauliBasis(Pi)
                evolve!(Wt, pb, theta)
                
                coeff_clip!(Wt, thresh=1.0e-12)
                e1 = expectation_value(W * Wt, ket)

                # --- POST pruning ---
                coeff_clip!(Wt, thresh=thresh)
                #WWt = W * Wt 
                e2 = expectation_value(W * Wt, ket)
                accumulated_error += e2-e1
            end           
            push!(snapshots, weight_profile(Wt))
            WWt = W * Wt #(at time interval)
            expval = expectation_value(WWt, ket) + accumulated_error # Contraction with reference ket
            Ctreal = real(expval) # Real part of C(t) = <O(0) * (U_i^ O U_i)>
            Ctimag = imag(expval)
            push!(rCtvals, Ctreal)
            push!(iCtvals, Ctimag)
    end

    tgrid = collect(range(0.0, stop=n_intervals*dt, length=length(rCtvals)))
    
    return rCtvals, iCtvals, tgrid, snapshots
end

# Define model parameters
Jx = 1.0
Jy = 1.0
Jz = 1.0
gx = 0.0
gy = 0.0
gz = 0.0
g = 0.00#-0.01
N = 6 #Total number of qubits
H = heisenberg_1D(N, Jx, Jy, Jz, x=gx)
@printf("1D-Heisenberg Hamiltonian (J= %.2f, Jz= %.2f): \n", Jx, Jz)
display(H)

# Define time evolution parameters
# Circuit divided in k layers
n_intervals = 100
t = 50.0 #Total Time Evolution
dt = t/n_intervals
threshold = 1e-6 #pruning threshold based on coeff.

# Define initial ket and Initial operator to be evolved under the circuit
ket = Ket(N,1)
o = Pauli(N, X=[3], Z=[1])
@printf("Initial operator O: %s \n", o.x)
o = PauliSum(o)
#o += Pauli(N, X=[2], Z=[4])
#o += Pauli(N, X=[1,2,3,4,5,6])

println("- - - Operator evolution - - - ")
println("Initial state |Psi0> ")
display(ket)
println("Initial operator O:")
display(o)

println("- - - Weight profile of H - - - ")
w_profile = weight_profile(H)
display(w_profile)

#
# * * * Perform the evolution
rRES, iRES, tgrid, w_snapshots = evolution_op(ket, o, H, n_intervals, dt; thresh=threshold)
println("- - - Pauli Weight Snapshots - - -")
for snapshot in w_snapshots
    println(snapshot)
end


#
#- - - Build DMD matrices
X = hcat(w_snapshots...)   # size (N+1) x T
#d = 50
#X = delay_embed(w_snapshots, d)
display(X)
res = fit_dmd(X; r=2)    # choose a small rank to start
print_dmd_summary(res; dt=dt)

using Plots

function plot_weight_heatmap(w_snapshots; tgrid=nothing)
    W = hcat(w_snapshots...)   # size: (N+1) × T
    nweights, nt = size(W)

    if tgrid === nothing
        tgrid = 0:nt-1
    end

    heatmap(
        tgrid,
        0:nweights-1,
        W,
        xlabel = "time",
        ylabel = "Pauli weight",
        title = "Pauli Weight Dynamics",
        legend = false
    )
end

function plot_weight_stack(w_snapshots; tgrid=nothing)
    W = hcat(w_snapshots...)
    nweights, nt = size(W)

    if tgrid === nothing
        tgrid = 0:nt-1
    end

    plt = plot(
        xlabel = "time",
        ylabel = "weight fraction",
        title = "Pauli Weight Distribution",
        legend = :right
    )

    for k in 1:nweights
        plot!(plt, tgrid, W[k, :], label = "k=$(k-1)", lw=2)
    end

    return plt
end

#function mean_weight(w::AbstractVector)
#    ks = 0:length(w)-1
#    return sum(ks .* w)
#end
#μs = [mean_weight(w) for w in w_snapshots]

#plot(tgrid, μs, xlabel="time", ylabel="⟨weight⟩", title="Mean Pauli Weight")
display(plot_weight_heatmap(w_snapshots; tgrid=tgrid))
#plot_weight_stack(w_snapshots; tgrid=tgrid)

function plot_dmd_modes(res::DMDResult; nmodes=4)
    r = min(nmodes, size(res.modes, 2))
    p = plot(
        xlabel = "Pauli weight",
        ylabel = "mode amplitude",
        title = "DMD modes in weight space",
        legend = :right
    )

    ks = 0:(size(res.modes, 1)-1)
    for j in 1:r
        plot!(p, ks, real(res.modes[:, j]), label = "mode $j",lw=2)
    end
    return p
end

function plot_dmd_modes_abs(res::DMDResult; nmodes=4)
    r = min(nmodes, size(res.modes, 2))
    p = plot(
        xlabel = "Pauli weight",
        ylabel = "|mode amplitude|",
        title = "Magnitude of DMD modes",
        legend = :right
    )

    ks = 0:(size(res.modes, 1)-1)
    for j in 1:r
        plot!(p, ks, abs.(res.modes[:, j]), label = "mode $j",lw=2)
        println("mode $j")
        println(res.modes[:,j])
    end
    return p
end

#plot_dmd_modes(res;nmodes=4)
display(plot_dmd_modes_abs(res;nmodes=4))