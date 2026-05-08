using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf
using Random

#"""
#The workflow is:
#
#- Build a Heisenberg Hamiltonian as a PauliSum.
#- Start from an initial Pauli operator O.
#- Evolve that operator under the Hamiltonian using repeated conjugations by Pauli rotations.
#- After each Trotter layer, record the operator’s weight profile: 
#  how much coefficient norm lives on Pauli strings of weight 0, 1, …, N.
#- Assemble those weight profiles into a snapshot matrix.
#- Run DMD on that matrix to extract dominant modes, eigenvalues, and a low-rank linear map.
#- Plot the resulting weight dynamics and DMD modes.
# ------------------------------------------------------------
#
#"""

# Small helpers
# ------------------------------------------------------------

coeff_clip!(ps; thresh=1e-16) = filter!(p -> abs(p.second) > thresh, ps)

function string_to_ket(bits::AbstractString)
    idx = 0
    for (i, ch) in enumerate(bits)
        if ch == '1'
            idx += 1 << (i - 1)
        end
    end
    return Ket{length(bits)}(idx), idx
end

neighbor(site::Int, N::Int) = site == N ? 1 : site + 1

# ------------------------------------------------------------
# Hamiltonian / Trotter construction
# ------------------------------------------------------------

function trott_unitary_sequence_Heisenberg(o::Union{Pauli{N}, PauliSum{N}};
    Jx=1.0, Jy=1.0, Jz=1.0, gx=0.1, gy=0.1, gz=0.1, k=10) where {N}

    generators = Pauli{N}[]
    parameters = Float64[]

    for _ in 1:k
        for site in 1:N
            nxt = neighbor(site, N)

            push!(generators, Pauli(N, X=[site, nxt]))
            push!(parameters, -Jx)

            push!(generators, Pauli(N, Y=[site, nxt]))
            push!(parameters, -Jy)
        end

        for site in 1:N
            nxt = neighbor(site, N)
            push!(generators, Pauli(N, Z=[site, nxt]))
            push!(parameters, -Jz)
        end

        if gx != 0.0
            for site in 1:N
                push!(generators, Pauli(N, X=[site]))
                push!(parameters, -gx)
            end
        end

        if gy != 0.0
            for site in 1:N
                push!(generators, Pauli(N, Y=[site]))
                push!(parameters, -gy)
            end
        end

        if gz != 0.0
            for site in 1:N
                push!(generators, Pauli(N, Z=[site]))
                push!(parameters, -gz)
            end
        end
    end

    return generators, parameters
end

function heisenberg_1D(N, Jx, Jy, Jz; x=0.0, y=0.0, z=0.0)
    H = PauliSum(N, Float64)

    for site in 1:N
        nxt = neighbor(site, N)
        H += -Jx * Pauli(N, X=[site, nxt])
        H += -Jy * Pauli(N, Y=[site, nxt])
        H += -Jz * Pauli(N, Z=[site, nxt])
    end

    for site in 1:N
        if x != 0.0
            H += x * Pauli(N, X=[site])
        end
        if y != 0.0
            H += y * Pauli(N, Y=[site])
        end
        if z != 0.0
            H += z * Pauli(N, Z=[site])
        end
    end

    return coeff_clip!(H)
end

# ------------------------------------------------------------
# Pauli propagation
# ------------------------------------------------------------

function evolve!(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N, T}
    cθ = cos(θ)
    sθ = 1im * sin(θ)
    added = PauliSum(N)

    for (p, c) in O
        if !PauliOperators.commute(p, G)
            tmp = c * sθ * G * p
            key = PauliBasis(tmp)
            added[key] = get(added, key, 0.0) + PauliOperators.coeff(tmp)
            O[p] *= cθ
        end
    end

    sum!(O, added)
    return O
end

function extract_hamiltonian_coeffs_and_ops(H::PauliSum{N, T}) where {N, T}
    ops = PauliBasis{N}[]
    coeffs = Float64[]
    for (p, c) in H
        push!(ops, p)
        push!(coeffs, float(c))
    end
    return ops, coeffs
end

# ------------------------------------------------------------
# Weight diagnostics
# ------------------------------------------------------------

weight(p::PauliBasis) = count_ones(p.x | p.z)

function weight_profile(W::PauliSum{N, T}) where {N, T}
    hist = zeros(Float64, N + 1)
    total = 0.0

    for (P, c) in W
        k = weight(P)
        a2 = abs2(c)
        hist[k + 1] += a2
        total += a2
    end

    if total > 0
        hist ./= total
    end

    return hist
end

function weight_stats(w::AbstractVector)
    ks = 0:length(w)-1
    μ = sum(ks .* w)
    σ2 = sum(((ks .- μ).^2) .* w)
    return μ, σ2
end

# ------------------------------------------------------------
# DMD
# ------------------------------------------------------------

struct DMDResult
    A_ls::Matrix{Float64}
    A_tilde::Matrix{ComplexF64}
    modes::Matrix{ComplexF64}
    evals::Vector{ComplexF64}
    amplitudes::Vector{ComplexF64}
    singular_values::Vector{Float64}
    residual_rel::Float64
end

function fit_dmd(X::AbstractMatrix{<:Real}; r::Union{Nothing,Int}=nothing, tol=1e-10)
    X1 = X[:, 1:end-1]
    X2 = X[:, 2:end]

    A_ls = Matrix(X2 * pinv(X1))
    residual_rel = norm(X2 - A_ls * X1) / max(norm(X2), eps())

    F = svd(X1; full=false)
    U, s, V = F.U, F.S, F.V

    if r === nothing
        r = max(count(>(tol * s[1]), s), 1)
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

    Φ = Matrix(X2 * Vr * Sinv * W)
    b = Φ \ complex.(X[:, 1])

    return DMDResult(A_ls, A_tilde, Φ, λ, b, s, residual_rel)
end

function print_dmd_summary(res::DMDResult; dt::Real=1.0, topk::Int=5)
    λ = res.evals
    growth = log.(abs.(λ)) ./ dt
    freq = angle.(λ) ./ dt
    idx = sortperm(abs.(λ), rev=true)

    println("---- DMD summary ----")
    println("relative LS residual = $(res.residual_rel)")
    println("numerical rank       = $(size(res.A_tilde, 1))")
    println("top singular values   = ", res.singular_values[1:min(end, topk)])

    println("\nDominant modes:")
    for j in 1:min(topk, length(idx))
        i = idx[j]
        println("  λ[$i] = $(λ[i])")
        println("      |λ| = $(abs(λ[i]))")
        println("      growth rate = $(growth[i])")
        println("      frequency    = $(freq[i])")
        println("      amplitude    = $(res.amplitudes[i])")
    end
end

function delay_embed(snapshots::Vector{<:AbstractVector}, q::Int)
    m = length(snapshots)
    d = length(snapshots[1])
    ncols = m - q + 1
    X = zeros(Float64, d * q, ncols)

    for k in 1:ncols
        for j in 1:q
            X[(j-1)*d + 1 : j*d, k] .= snapshots[k + j - 1]
        end
    end

    return X
end

function dmd_bootstrap(snapshots::Vector{<:AbstractVector}; B::Int=100, r::Union{Nothing,Int}=nothing, seed::Int=1)
    rng = MersenneTwister(seed)
    m = length(snapshots)
    evals = Vector{Vector{ComplexF64}}(undef, B)

    for b in 1:B
        lo = rand(rng, 1:m-3)
        hi = rand(rng, lo+2:m)
        Xb = hcat(snapshots[lo:hi]...)
        evals[b] = fit_dmd(Xb; r=r).evals
    end

    return evals
end

# ------------------------------------------------------------
# Main evolution loop
# ------------------------------------------------------------

function evolution_op(ket, o::PauliSum{N, T}, H::PauliSum{N, T}, n_intervals, dt;
    thresh::Float64=1e-3) where {N, T}

    O0 = deepcopy(o)
    Ot = deepcopy(o)

    corr_real = Float64[]
    corr_imag = Float64[]
    snapshots = Vector{Vector{Float64}}()

    push!(snapshots, weight_profile(Ot))

    c0 = expectation_value(O0 * Ot, ket)
    push!(corr_real, real(c0))
    push!(corr_imag, imag(c0))

    generators, angles = extract_hamiltonian_coeffs_and_ops(H)
    nt = length(angles)

    println("Total Rotations: ", nt * n_intervals)

    for _ in 1:n_intervals
        accumulated_error = 0.0 + 0.0im

        for j in 1:nt
            θ = 2 * dt * angles[j]
            evolve!(Ot, generators[j], θ)

            coeff_clip!(Ot; thresh=1e-12)
            before = expectation_value(O0 * Ot, ket)

            coeff_clip!(Ot; thresh=thresh)
            after = expectation_value(O0 * Ot, ket)

            accumulated_error += after - before
        end

        push!(snapshots, weight_profile(Ot))

        c = expectation_value(O0 * Ot, ket) + accumulated_error
        push!(corr_real, real(c))
        push!(corr_imag, imag(c))
    end

    tgrid = collect(range(0.0, stop=n_intervals * dt, length=length(corr_real)))
    return corr_real, corr_imag, tgrid, snapshots
end

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

function plot_weight_heatmap(w_snapshots; tgrid=nothing)
    W = hcat(w_snapshots...)
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

function plot_dmd_modes_abs(res::DMDResult; nmodes=4)
    r = min(nmodes, size(res.modes, 2))
    ks = 0:(size(res.modes, 1)-1)

    p = plot(
        xlabel = "Pauli weight",
        ylabel = "|mode amplitude|",
        title = "Magnitude of DMD modes",
        legend = :right
    )

    for j in 1:r
        plot!(p, ks, abs.(res.modes[:, j]), label = "mode $j", lw=2)
    end

    return p
end

#=
 Questions to address:
 Is the weight dynamics low-dimensional?
 Does DMD find a few dominant temporal modes?
 Do those modes line up with early / mid / late scrambling?
 Does pruning change the modes in a way that predicts error?
=#
# Experiment script for studying DMD + Pauli propagation under Heisenberg evolution.
#
# Assumes the following functions/types are already available in the session:
#   - heisenberg_1D
#   - evolution_op
#   - fit_dmd
#   - weight_stats
#   - plot_weight_heatmap
#   - plot_dmd_modes_abs
#   - coeff_clip! (optional, used inside evolution_op)
#
# This script runs a threshold sweep, compares each run to a reference baseline,
# and computes a small set of diagnostics intended to address:
#   1) low-dimensional structure of weight distributions
#   2) dominant temporal patterns via DMD
#   3) physical interpretation of modes
#   4) pruning-error diagnosis

mean_weight(w::AbstractVector) = sum((0:length(w)-1) .* w)

function weight_entropy(w::AbstractVector; ϵ::Float64=1e-15)
    p = clamp.(w, ϵ, 1.0)
    return -sum(p .* log.(p))
end

function l1_distance(a::AbstractVector, b::AbstractVector)
    return sum(abs.(a .- b))
end

function l2_distance(a::AbstractVector, b::AbstractVector)
    return norm(a .- b)
end

function relative_l2_distance(a::AbstractVector, b::AbstractVector)
    denom = max(norm(b), eps())
    return norm(a .- b) / denom
end

function dominant_rank(s::AbstractVector{<:Real}; energy::Float64=0.95)
    tot = sum(abs2, s)
    if tot <= 0
        return 0
    end
    acc = 0.0
    for (i, σ) in enumerate(s)
        acc += abs2(σ)
        if acc / tot >= energy
            return i
        end
    end
    return length(s)
end

function time_series_metrics(w_snapshots::Vector{<:AbstractVector})
    nt = length(w_snapshots)
    μ = zeros(Float64, nt)
    σ2 = zeros(Float64, nt)
    ent = zeros(Float64, nt)

    for t in 1:nt
        μ[t], σ2[t] = weight_stats(w_snapshots[t])
        ent[t] = weight_entropy(w_snapshots[t])
    end

    return (
        mean_weight = μ,
        variance = σ2,
        entropy = ent,
    )
end

function fit_dmd_from_snapshots(w_snapshots::Vector{<:AbstractVector}; r::Union{Nothing,Int}=nothing)
    X = hcat(w_snapshots...)
    return fit_dmd(X; r=r)
end

# ------------------------------------------------------------
# Data container
# ------------------------------------------------------------

struct SweepRun
    threshold::Float64
    rRES::Vector{Float64}
    iRES::Vector{Float64}
    tgrid::Vector{Float64}
    w_snapshots::Vector{Vector{Float64}}
    X::Matrix{Float64}
    dmd::DMDResult
    mean_weight::Vector{Float64}
    variance::Vector{Float64}
    entropy::Vector{Float64}
    term_count::Union{Nothing,Vector{Int}}
end

# ------------------------------------------------------------
# Main experiment driver
# ------------------------------------------------------------

function run_threshold_sweep(ket, o, H, n_intervals, dt, thresholds::AbstractVector;
    dmd_rank::Union{Nothing,Int}=2,
    verbose::Bool=true)

    results = Dict{Float64, SweepRun}()

    for τ in thresholds
        if verbose
            @printf("\n=== Running threshold = %.2e ===\n", τ)
        end

        rRES, iRES, tgrid, w_snapshots = evolution_op(ket, o, H, n_intervals, dt; thresh=τ)
        X = hcat(w_snapshots...)
        #d = 10 #delay dimension
        #X = delay_embed(w_snapshots, d)
 
        dmd = fit_dmd(X; r=dmd_rank)
        metrics = time_series_metrics(w_snapshots)

        results[τ] = SweepRun(
            τ,
            rRES,
            iRES,
            tgrid,
            w_snapshots,
            X,
            dmd,
            metrics.mean_weight,
            metrics.variance,
            metrics.entropy,
            nothing,
        )

        if verbose
            @printf("  DMD residual: %.4e\n", dmd.residual_rel)
            @printf("  singular values kept: %d\n", size(dmd.A_tilde, 1))
            @printf("  first singular values: %s\n", string(dmd.singular_values[1:min(end, 5)]))
        end
    end

    return results
end

function compare_to_baseline(results::Dict{Float64, SweepRun}, baseline_threshold::Float64)
    @assert haskey(results, baseline_threshold) "Baseline threshold not found in results."

    ref = results[baseline_threshold]
    out = Dict{Float64, Dict{Symbol, Any}}()

    for (τ, run) in results
        nt = min(length(run.w_snapshots), length(ref.w_snapshots))
        err_l1 = zeros(Float64, nt)
        err_l2 = zeros(Float64, nt)
        err_mu = zeros(Float64, nt)
        err_var = zeros(Float64, nt)
        err_r = zeros(Float64, nt)
        err_i = zeros(Float64, nt)

        for t in 1:nt
            err_l1[t] = l1_distance(run.w_snapshots[t], ref.w_snapshots[t])
            err_l2[t] = l2_distance(run.w_snapshots[t], ref.w_snapshots[t])
            err_mu[t] = abs(run.mean_weight[t] - ref.mean_weight[t])
            err_var[t] = abs(run.variance[t] - ref.variance[t])
            err_r[t] = abs(run.rRES[t] - ref.rRES[t])
            err_i[t] = abs(run.iRES[t] - ref.iRES[t])
        end

        out[τ] = Dict(
            :err_l1 => err_l1,
            :err_l2 => err_l2,
            :err_mu => err_mu,
            :err_var => err_var,
            :err_r => err_r,
            :err_i => err_i,
            :final_l1 => err_l1[end],
            :final_l2 => err_l2[end],
            :max_l1 => maximum(err_l1),
            :max_l2 => maximum(err_l2),
            :final_mu => err_mu[end],
            :max_mu => maximum(err_mu),
            :final_r => err_r[end],
            :final_i => err_i[end],
        )
    end

    return out
end

function summarize_sweep(results::Dict{Float64, SweepRun}, baseline_threshold::Float64)
    comp = compare_to_baseline(results, baseline_threshold)
    thresholds = sort(collect(keys(results)))

    println("\n================ SWEEP SUMMARY ================")
    println("Baseline threshold: ", baseline_threshold)

    for τ in thresholds
        run = results[τ]
        err = comp[τ]
        rank95 = dominant_rank(run.dmd.singular_values; energy=0.95)
        rank99 = dominant_rank(run.dmd.singular_values; energy=0.99)

        @printf("\nThreshold = %.2e\n", τ)
        @printf("  DMD residual        : %.4e\n", run.dmd.residual_rel)
        @printf("  rank for 95%% energy : %d\n", rank95)
        @printf("  rank for 99%% energy : %d\n", rank99)
        @printf("  final L1 error      : %.4e\n", err[:final_l1])
        @printf("  max L1 error        : %.4e\n", err[:max_l1])
        @printf("  final mean-weight err: %.4e\n", err[:final_mu])
        @printf("  final correlator err : %.4e\n", err[:final_r])
    end

    return comp
end

# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------

function plot_threshold_errors(results::Dict{Float64, SweepRun}, comp::Dict{Float64, Dict{Symbol, Any}};
    baseline_threshold::Float64)

    thresholds = sort(collect(keys(results)))

    p1 = plot(title="Final L1 error vs threshold", xlabel="threshold", ylabel="final L1 error", xscale=:log10, yscale=:log10)
    p2 = plot(title="Final correlator error vs threshold", xlabel="threshold", ylabel="final |ΔC(t_final)|", xscale=:log10, yscale=:log10)
    p3 = plot(title="DMD residual vs threshold", xlabel="threshold", ylabel="DMD residual", xscale=:log10, yscale=:log10)

    for τ in thresholds
        run = results[τ]
        err = comp[τ]
        scatter!(p1, [τ], [max(err[:final_l1], eps())], label=false)
        scatter!(p2, [τ], [max(err[:final_r], eps())], label=false)
        scatter!(p3, [τ], [max(run.dmd.residual_rel, eps())], label=false)
    end

    vline!(p1, [baseline_threshold], label="baseline", linestyle=:dash)
    vline!(p2, [baseline_threshold], label="baseline", linestyle=:dash)
    vline!(p3, [baseline_threshold], label="baseline", linestyle=:dash)

    return p1, p2, p3
end

function plot_time_traces(results::Dict{Float64, SweepRun}, baseline_threshold::Float64; quantity::Symbol=:mean_weight)
    ref = results[baseline_threshold]
    thresholds = sort(collect(keys(results)))

    p = plot(
        xlabel = "time",
        ylabel = string(quantity),
        title = "Time traces across thresholds",
        legend = :right,
    )

    ref_vals = getproperty(ref, quantity)
    plot!(p, ref.tgrid, ref_vals, label = "baseline $(baseline_threshold)", lw=3)

    for τ in thresholds
        if τ == baseline_threshold
            continue
        end
        run = results[τ]
        vals = getproperty(run, quantity)
        plot!(p, run.tgrid, vals, label = string(τ), lw=1.5, alpha=0.8)
    end

    return p
end

function plot_error_traces(results::Dict{Float64, SweepRun}, comp::Dict{Float64, Dict{Symbol, Any}};
    baseline_threshold::Float64, quantity::Symbol=:err_l1)

    ref = results[baseline_threshold]
    thresholds = sort(collect(keys(results)))

    p = plot(
        xlabel = "time",
        ylabel = string(quantity),
        title = "Error traces vs baseline",
        legend = :right,
    )

    for τ in thresholds
        if τ == baseline_threshold
            continue
        end
        err = comp[τ][quantity]
        plot!(p, ref.tgrid[1:length(err)], err, label = string(τ), lw=2)
    end

    return p
end

function plot_singular_values(results::Dict{Float64, SweepRun})
    thresholds = sort(collect(keys(results)))
    p = plot(
        xlabel = "mode index",
        ylabel = "singular value",
        title = "Singular value spectra",
        yscale = :log10,
        legend = :right,
    )

    for τ in thresholds
        s = results[τ].dmd.singular_values
        plot!(p, 1:length(s), s, label = string(τ), lw=2)
    end

    return p
end

function plot_mean_weight_and_entropy(results::Dict{Float64, SweepRun}, baseline_threshold::Float64)
    ref = results[baseline_threshold]
    thresholds = sort(collect(keys(results)))

    p1 = plot(xlabel="time", ylabel="mean weight", title="Mean Pauli weight", legend=:right)
    p2 = plot(xlabel="time", ylabel="entropy", title="Weight entropy", legend=:right)

    plot!(p1, ref.tgrid, ref.mean_weight, label="baseline $(baseline_threshold)", lw=3)
    plot!(p2, ref.tgrid, ref.entropy, label="baseline $(baseline_threshold)", lw=3)

    for τ in thresholds
        if τ == baseline_threshold
            continue
        end
        run = results[τ]
        plot!(p1, run.tgrid, run.mean_weight, label=string(τ), lw=1.5, alpha=0.8)
        plot!(p2, run.tgrid, run.entropy, label=string(τ), lw=1.5, alpha=0.8)
    end

    return p1, p2
end

# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

function main_experiment()
    # Physical setup
    Jx = 1.0
    Jy = 1.0
    Jz = 1.0
    gx = 0.0
    gy = 0.0
    gz = 0.0
    N = 6

    # Evolution setup
    n_intervals = 100
    total_time = 50.0
    dt = total_time / n_intervals

    # Initial state/operator
    ket = Ket(N, 1)
    o = Pauli(N, X=[3], Z=[1])
    o = PauliSum(o)

    # Hamiltonian
    H = heisenberg_1D(N, Jx, Jy, Jz; x=gx, y=gy, z=gz)

    # Threshold sweep
    thresholds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    baseline_threshold = minimum(thresholds)

    println("Running threshold sweep...")
    results = run_threshold_sweep(ket, o, H, n_intervals, dt, thresholds; dmd_rank=2, verbose=true)

    println("Comparing against baseline...")
    comp = summarize_sweep(results, baseline_threshold)

    # Plots
    p_err1, p_err2, p_err3 = plot_threshold_errors(results, comp; baseline_threshold=baseline_threshold)
    p_mu, p_ent = plot_mean_weight_and_entropy(results, baseline_threshold)
    p_sv = plot_singular_values(results)

    display(p_err1)
    display(p_err2)
    display(p_err3)
    display(p_mu)
    display(p_ent)
    display(p_sv)

    # Baseline-specific diagnostics
    ref = results[baseline_threshold]
    display(plot_weight_heatmap(ref.w_snapshots; tgrid=ref.tgrid))
    display(plot_dmd_modes_abs(ref.dmd; nmodes=4))

    return results, comp
end

# Uncomment to run immediately in a script context:
results, comp = main_experiment();

#println("* * * RESULTS * * *")
#println(results)
#println("Comp:", comp)

