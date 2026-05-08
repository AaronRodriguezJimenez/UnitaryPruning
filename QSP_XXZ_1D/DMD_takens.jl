# Takens embedding + DMD validation script
#
# Purpose:
#   1) Verify that the delay embedding is constructed correctly.
#   2) Check whether DMD on delay-embedded data improves fit/prediction as q changes.
#   3) Distinguish between a correct embedding, a correct DMD fit, and a plotting/interpretation issue.
#
# Assumes these definitions are available in the session (from your PP code):
#   - delay_embed
#   - fit_dmd
#   - evolution_op
#   - heisenberg_1D
#   - weight_stats
#   - plot_weight_heatmap
#
# This script is intentionally self-contained in its analysis logic.

using LinearAlgebra
using Statistics
using Printf
using Random
using Plots

# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

mean_weight(w::AbstractVector) = sum((0:length(w)-1) .* w)

function l2_distance(a::AbstractVector, b::AbstractVector)
    return norm(a .- b)
end

function relative_l2_distance(a::AbstractVector, b::AbstractVector)
    denom = max(norm(b), eps())
    return norm(a .- b) / denom
end

function time_series_metrics(w_snapshots::Vector{<:AbstractVector})
    nt = length(w_snapshots)
    μ = zeros(Float64, nt)
    σ2 = zeros(Float64, nt)
    for t in 1:nt
        μ[t], σ2[t] = weight_stats(w_snapshots[t])
    end
    return μ, σ2
end

# ------------------------------------------------------------
# Delay-embedding helpers
# ------------------------------------------------------------

"""
Construct a Takens-style delay embedding from a list of snapshot vectors.
Each column is [x_t; x_{t+1}; ...; x_{t+q-1}].
"""
function delay_embed_checked(snapshots::Vector{<:AbstractVector}, q::Int)
    m = length(snapshots)
    d = length(snapshots[1])
    @assert q >= 1 "q must be at least 1"
    @assert m >= q "need at least q snapshots"

    for s in snapshots
        @assert length(s) == d "all snapshots must have the same dimension"
    end

    ncols = m - q + 1
    X = zeros(Float64, d * q, ncols)

    for k in 1:ncols
        for j in 1:q
            X[(j-1)*d + 1 : j*d, k] .= snapshots[k + j - 1]
        end
    end

    return X
end

embed_snapshots_checked(snapshots::Vector{<:AbstractVector}; q::Int=1) = q == 1 ? hcat(snapshots...) : delay_embed_checked(snapshots, q)

function extract_blocks(v::AbstractVector, d::Int, q::Int)
    @assert length(v) == d * q
    return [v[(j-1)*d + 1 : j*d] for j in 1:q]
end

# ------------------------------------------------------------
# DMD forecast helpers
# ------------------------------------------------------------

"""
Fit DMD to X = [x_1, x_2, ..., x_m] and forecast one step ahead.
Returns the next-step predicted snapshot matrix in the same coordinate system.
"""
function dmd_forecast_matrix(X::AbstractMatrix; r::Union{Nothing,Int}=nothing)
    res = fit_dmd(X; r=r)
    A = res.A_ls
    Xpred = A * X[:, 1:end-1]
    return res, Xpred
end

function reconstruction_error(X::AbstractMatrix; r::Union{Nothing,Int}=nothing)
    res = fit_dmd(X; r=r)
    X1 = X[:, 1:end-1]
    X2 = X[:, 2:end]
    X2hat = res.A_ls * X1
    rel = norm(X2 - X2hat) / max(norm(X2), eps())
    return res, rel
end

"""
For delay-embedded data, compare the predicted embedded vector against the true next embedded vector.
The error is computed in the augmented delay space.
"""
function embedded_forecast_error(snapshots::Vector{<:AbstractVector}; q::Int=1, r::Union{Nothing,Int}=nothing)
    X = embed_snapshots_checked(snapshots; q=q)
    res = fit_dmd(X; r=r)

    # X[:, 1:end-1] corresponds to delay windows starting at t=1,...,m-q
    # X[:, 2:end]   corresponds to the true next embedded windows
    X1 = X[:, 1:end-1]
    X2 = X[:, 2:end]
    X2hat = res.A_ls * X1
    rel = norm(X2 - X2hat) / max(norm(X2), eps())

    return res, rel
end

"""
For delay embedding, the first block of each embedded column is the physically relevant current snapshot.
This helper checks that the first block matches the original snapshots used to build the embedding.
"""
function embedding_consistency_check(snapshots::Vector{<:AbstractVector}, q::Int)
    X = embed_snapshots_checked(snapshots; q=q)
    d = length(snapshots[1])
    ncols = size(X, 2)

    max_err = 0.0
    for k in 1:ncols
        first_block = X[1:d, k]
        max_err = max(max_err, norm(first_block - snapshots[k]))
    end
    return max_err
end

# ------------------------------------------------------------
# Synthetic validation case
# ------------------------------------------------------------

"""
A simple low-dimensional oscillatory signal with two frequencies.
This is useful for checking that delay embedding can improve DMD rank capture.
"""
function synthetic_signal(m::Int; ω1::Float64=0.12, ω2::Float64=0.31)
    t = collect(0:m-1)
    x1 = sin.(ω1 .* t) .+ 0.3 .* cos.(ω2 .* t)
    x2 = cos.(ω1 .* t) .- 0.2 .* sin.(ω2 .* t)
    x3 = 0.5 .* sin.(0.5 .* ω1 .* t .+ 0.1)
    return [Float64[x1[i], x2[i], x3[i]] for i in 1:m]
end

function validate_embedding_on_synthetic(; m::Int=200, q_values=1:12, r::Union{Nothing,Int}=2)
    snapshots = synthetic_signal(m)
    results = Dict{Int, Dict{Symbol, Any}}()

    for q in q_values
        X = embed_snapshots_checked(snapshots; q=q)
        res = fit_dmd(X; r=r)
        rel = norm(X[:, 2:end] - res.A_ls * X[:, 1:end-1]) / max(norm(X[:, 2:end]), eps())
        results[q] = Dict(
            :res => res,
            :rel_fit => rel,
            :rank95 => begin
                s = res.singular_values
                tot = sum(abs2, s)
                acc = 0.0
                rr = length(s)
                for (i, σ) in enumerate(s)
                    acc += abs2(σ)
                    if acc / tot >= 0.95
                        rr = i
                        break
                    end
                end
                rr
            end,
        )
    end

    return snapshots, results
end

function plot_synthetic_q_sweep(results::Dict{Int, Dict{Symbol, Any}})
    qs = sort(collect(keys(results)))
    fit_err = [results[q][:rel_fit] for q in qs]
    rank95 = [results[q][:rank95] for q in qs]

    p1 = plot(qs, fit_err, marker=:circle, xlabel="embedding q", ylabel="relative fit error", title="DMD fit error vs embedding dimension", legend=false, yscale=:log10)
    p2 = plot(qs, rank95, marker=:circle, xlabel="embedding q", ylabel="rank for 95% energy", title="Effective rank vs embedding dimension", legend=false)
    return p1, p2
end

# ------------------------------------------------------------
# PP validation case
# ------------------------------------------------------------

function run_pp_validation(ket, o, H, n_intervals::Int, dt::Real;
    threshold::Float64=1e-10,
    q_values = 1:10,
    dmd_rank::Union{Nothing,Int}=2)

    rRES, iRES, tgrid, w_snapshots = evolution_op(ket, o, H, n_intervals, dt; thresh=threshold)
    d = length(w_snapshots[1])

    checks = Dict{Int, Dict{Symbol, Any}}()

    for q in q_values
        @assert q <= length(w_snapshots) "q=$q is too large for the number of snapshots"
        X = embed_snapshots_checked(w_snapshots; q=q)
        res = fit_dmd(X; r=dmd_rank)

        X1 = X[:, 1:end-1]
        X2 = X[:, 2:end]
        rel_fit = norm(X2 - res.A_ls * X1) / max(norm(X2), eps())
        consistency = embedding_consistency_check(w_snapshots, q)

        checks[q] = Dict(
            :res => res,
            :rel_fit => rel_fit,
            :consistency => consistency,
            :singular_values => res.singular_values,
            :rank95 => begin
                s = res.singular_values
                tot = sum(abs2, s)
                acc = 0.0
                rr = length(s)
                for (i, σ) in enumerate(s)
                    acc += abs2(σ)
                    if acc / tot >= 0.95
                        rr = i
                        break
                    end
                end
                rr
            end,
            :X => X,
        )
    end

    return (
        rRES = rRES,
        iRES = iRES,
        tgrid = tgrid,
        w_snapshots = w_snapshots,
        d = d,
        checks = checks,
    )
end

function plot_pp_q_sweep(checks::Dict{Int, Dict{Symbol, Any}})
    qs = sort(collect(keys(checks)))
    fit_err = [checks[q][:rel_fit] for q in qs]
    rank95 = [checks[q][:rank95] for q in qs]
    consistency = [checks[q][:consistency] for q in qs]

    p1 = plot(qs, fit_err, marker=:circle, xlabel="embedding q", ylabel="relative fit error", title="PP DMD fit error vs embedding dimension", legend=false, yscale=:log10)
    p2 = plot(qs, rank95, marker=:circle, xlabel="embedding q", ylabel="rank for 95% energy", title="PP effective rank vs embedding dimension", legend=false)
    p3 = plot(qs, consistency, marker=:circle, xlabel="embedding q", ylabel="max embedding consistency error", title="Embedding consistency check", legend=false, yscale=:log10)
    return p1, p2, p3
end

function inspect_one_q(checks::Dict{Int, Dict{Symbol, Any}}, q::Int; nmodes::Int=4)
    @assert haskey(checks, q) "q=$q not found in checks"
    res = checks[q][:res]
    X = checks[q][:X]
    dmd_rank = size(res.A_tilde, 1)

    println("--- q = $q ---")
    println("embedded matrix size = ", size(X))
    println("DMD rank             = ", dmd_rank)
    println("relative fit error    = ", checks[q][:rel_fit])
    println("consistency error     = ", checks[q][:consistency])
    println("leading singular vals = ", res.singular_values[1:min(end, 5)])

    p = plot(
        xlabel = "embedded coordinate index",
        ylabel = "|mode amplitude|",
        title = "DMD modes in delay space for q=$q",
        legend = :right,
    )
    ks = 1:size(res.modes, 1)
    for j in 1:min(nmodes, size(res.modes, 2))
        plot!(p, ks, abs.(res.modes[:, j]), label = "mode $j", lw=2)
    end

    return p
end

# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

function main()
    println("==============================")
    println("Takens + DMD validation suite")
    println("==============================")

    # 1) Synthetic test
    println("\n[1] Synthetic validation")
    _, syn = validate_embedding_on_synthetic(m=200, q_values=1:12, r=2)
    p_syn1, p_syn2 = plot_synthetic_q_sweep(syn)
    display(p_syn1)
    display(p_syn2)

    # 2) PP test (assumes your PP objects/functions exist in session)
    println("\n[2] PP validation")
    # Example setup; replace with your current values if needed.
    N = 6
    ket = Ket(N, 1)
    o = PauliSum(Pauli(N, X=[3], Z=[1]))
    H = heisenberg_1D(N, 1.0, 1.0, 1.0; x=0.0, y=0.0, z=0.0)

    pp = run_pp_validation(ket, o, H, 100, 0.5; threshold=1e-10, q_values=1:10, dmd_rank=2)
    p1, p2, p3 = plot_pp_q_sweep(pp.checks)
    display(p1)
    display(p2)
    display(p3)

    # Look at one representative embedding dimension
    qstar = 5
    display(inspect_one_q(pp.checks, qstar; nmodes=4))

    return (synthetic = syn, pp = pp)
end

# Uncomment to run immediately:
results = main()