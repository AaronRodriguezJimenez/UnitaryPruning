using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf
using Random
using StatsBase

function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
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

"""
    qdrift_propagator(H::PauliSum, t::Real, eps::Real;
                      seed::Union{Int,Nothing}=nothing,
                      plot::Bool=true, print_selection::Bool=true)

Perform a qDRIFT approximation for e^{i H t} where H is a PauliSum.

Returns a NamedTuple with:
  V         = simulated qDRIFT unitary (matrix)
  U_exact   = exact unitary exp(i * t * H) (matrix)
  sample    = vector of sampled indices (1-based)
  probs     = sampling distribution p_j
  λ         = sum(abs(h_j))
  τ         = per-step angle t*λ/N
  N         = number of samples used
  ops, coeffs = original lists of PauliBasis and coefficients
"""
function qdrift_propagator(H::PauliSum{N,T}, t::Real, eps::Real;
                           seed::Union{Int,Nothing}=nothing,
                           plot::Bool=true, print_selection::Bool=true) where {N,T}

    # Extract Pauli strings and coefficients
    ops, coeffs = extract_hamiltonian_coeffs_and_ops(H)
    @printf("Hamiltonian has %.2f terms \n", length(coeffs))
    L = length(coeffs)
    if L == 0
        error("Hamiltonian has no terms.")
    end

    # λ = sum_j |h_j| and probabilities p_j = |h_j| / λ
    λ = sum(abs.(coeffs))
    if λ == 0.0
        error("Sum of absolute coefficients is zero (λ=0).")
    end
    probs = abs.(coeffs) ./ λ

    # Number of samples (N); 
    Nsamples = ceil(Int, 2 * (λ * t)^2 / eps)
    if Nsamples < 1
        Nsamples = 1
    end

    # Set RNG seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # i.i.d. sampling of indices 1..L with replacement according to probs
    sample_list = sample(1:L, Weights(probs), Nsamples; replace=true)

    # Per-step angle tau = t * λ / N
    τ = t * λ / Nsamples

    # Helper: convert PauliBasis -> dense matrix
    pauli_to_matrix(op::PauliBasis{K}) where {K} = Matrix(op)

    # Build full Hamiltonian matrix H_full = sum_j h_j H_j (dense)
    dim = 2^N
    H_full = zeros(ComplexF64, dim, dim)
    for (op, c) in zip(ops, coeffs)
        H_full .+= c * pauli_to_matrix(op)
    end

    # Build qDRIFT simulated unitary V = U_N ... U_1
    V = Matrix{ComplexF64}(I, dim, dim)
    for idx in sample_list
        op = ops[idx]
        H_j = pauli_to_matrix(op)          # assumed normalized (||H_j||=1 for Pauli strings)
        s = sign(coeffs[idx])              # sign(h_j)
        Uj = exp(1im * τ * s * H_j)        # exp(i τ sign(h_j) H_j)
        V = Uj * V                         # left-multiply to respect order
    end

    # Exact unitary for comparison
    U_exact = exp(1im * t * H_full)

    # Print selection if requested
    if print_selection
        println("=== qDRIFT Selection (first 200 shown) ===")
        println(sample_list[1:min(end,200)])
        println("... (total samples = $Nsamples)")
    end

    # Plot distribution if requested
    if plot
        plt_bar = bar!(probs, legend=false, xlabel="Term index j", ylabel="p_j",
            title = @sprintf("qDRIFT sampling distribution (λ = %.6g)", λ))
        savefig(plt_bar, "qdrift_sampling_distribution.pdf")
    end

    return (V=V, U_exact=U_exact, sample=sample_list, probs=probs,
            λ=λ, τ=τ, N=Nsamples, ops=ops, coeffs=coeffs)
end

# demo parameters
Jx, Jy, Jz = 1.0, 1.0, 2.0
Nqubits = 4
H = heisenberg_1D(Nqubits, Jx, Jy, Jz)
t = 1.0
eps = 0.1


res = qdrift_propagator(H, t, eps; seed=666, plot=true, print_selection=true)

# quick diagnostics
diff_norm = opnorm(res.V - res.U_exact, 2)
@printf("Operator norm difference ||V - e^{iHt}|| = %.6e  (N=%d, τ=%.6g, λ=%.6g)\n",
        diff_norm, res.N, res.τ, res.λ)

@printf("Hamiltonian after qDRIFT has %.2f terms \n", length(res.coeffs))
