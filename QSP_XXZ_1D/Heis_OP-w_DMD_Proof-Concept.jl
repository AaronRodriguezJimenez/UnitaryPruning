using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf
using Random

# ============================================================
# Takens-DMD proof of concept for operator-weight dynamics
# ============================================================
# This script:
#   1) builds a 1D Heisenberg Hamiltonian,
#   2) evolves an initial local Pauli operator,
#   3) records normalized Pauli-weight snapshots,
#   4) performs Takens-style delay embedding,
#   5) fits DMD in embedded space,
#   6) reconstructs the trajectory,
#   7) compares original vs reconstructed weight dynamics.
# ============================================================

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

coeff_clip!(ps; thresh=1e-16) = filter!(p -> abs(p.second) > thresh, ps)
neighbor(site::Int, N::Int) = site == N ? 1 : site + 1

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

mean_weight_series(snapshots::Vector{<:AbstractVector}) = [weight_stats(w)[1] for w in snapshots]

function stack_snapshots(snapshots::Vector{<:AbstractVector})
    return hcat(snapshots...)
end

# ------------------------------------------------------------
# Hamiltonian
# ------------------------------------------------------------

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

function evolve_snapshots(ket, o::PauliSum{N, T}, H::PauliSum{N, T}, n_intervals, dt;
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

    @printf("Total Pauli rotations: %d\n", nt * n_intervals)

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

function fit_dmd(X::AbstractMatrix{<:Real}; r::Union{Nothing,Int}=nothing, tol=1e-10)
    @assert size(X, 2) >= 2 "Need at least two columns for DMD."

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
    println("effective rank       = $(size(res.A_tilde, 1))")
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

# ------------------------------------------------------------
# Takens delay embedding
# ------------------------------------------------------------

"""
Takens-style delay embedding.
Each column is [x_t; x_{t+1}; ...; x_{t+q-1}].
"""
function delay_embed(snapshots::Vector{<:AbstractVector}, q::Int)
    m = length(snapshots)
    d = length(snapshots[1])
    @assert q >= 1 "q must be at least 1"
    @assert m >= q "need at least q snapshots"

    ncols = m - q + 1
    X = zeros(Float64, d * q, ncols)

    for k in 1:ncols
        for j in 1:q
            X[(j - 1) * d + 1 : j * d, k] .= snapshots[k + j - 1]
        end
    end

    return X
end

embed_snapshots(snapshots::Vector{<:AbstractVector}; q::Int=1) = q == 1 ? stack_snapshots(snapshots) : delay_embed(snapshots, q)

function extract_physical_block(Xemb::AbstractMatrix, d::Int)
    return Xemb[1:d, :]
end

function reconstruct_dmd(res::DMDResult, nt::Int)
    Xhat = zeros(ComplexF64, size(res.modes, 1), nt)

    for k in 0:nt-1
        zk = res.amplitudes .* (res.evals .^ k)
        Xhat[:, k + 1] .= res.modes * zk
    end

    return real.(Xhat)
end

function embed_fit_summary(snapshots::Vector{<:AbstractVector}; q::Int=1, r::Union{Nothing,Int}=2)
    X = embed_snapshots(snapshots; q=q)
    res = fit_dmd(X; r=r)
    X1 = X[:, 1:end-1]
    X2 = X[:, 2:end]
    rel_fit = norm(X2 - res.A_ls * X1) / max(norm(X2), eps())
    rank95 = dominant_rank(res.singular_values; energy=0.95)
    rank99 = dominant_rank(res.singular_values; energy=0.99)

    return (
        X = X,
        res = res,
        rel_fit = rel_fit,
        rank95 = rank95,
        rank99 = rank99,
    )
end

"""
 Reconstruction of the physical snapshots by averaging over the relevant blocks in the embedded DMD reconstruction.
"""
function unembed_physical_average(Xemb::AbstractMatrix, d::Int, q::Int, m::Int)
    ncols = size(Xemb, 2)
    @assert ncols == m - q + 1 "Embedded matrix has inconsistent number of columns."

    snapshots = Vector{Vector{Float64}}(undef, m)

    for t in 1:m
        accum = zeros(Float64, d)
        count = 0

        kmin = max(1, t - q + 1)
        kmax = min(t, ncols)

        for k in kmin:kmax
            j = t - k + 1
            block = Xemb[(j - 1) * d + 1 : j * d, k]
            accum .+= real.(block)
            count += 1
        end

        snapshots[t] = accum ./ max(count, 1)
    end

    return snapshots
end

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

function plot_weight_heatmap(snapshots::Vector{<:AbstractVector}; tgrid=nothing, title_str="Pauli Weight Dynamics")
    W = stack_snapshots(snapshots)
    if tgrid === nothing
        tgrid = 0:size(W, 2)-1
    end
    @assert length(tgrid) == size(W, 2) "tgrid length must match the number of snapshot columns."

    heatmap(
        tgrid,
        0:size(W, 1)-1,
        W,
        xlabel="time",
        ylabel="Pauli weight",
        title=title_str,
    )
end

function plot_dmd_singular_values(res::DMDResult)
    plot(
        1:length(res.singular_values),
        res.singular_values,
        yscale=:log10,
        xlabel="mode index",
        ylabel="singular value",
        title="DMD singular value spectrum",
        lw=2,
    )
end

function plot_mean_weight(tgrid, snapshots_true::Vector{<:AbstractVector}, snapshots_dmd::Vector{<:AbstractVector})
    nt = min(length(tgrid), length(snapshots_true), length(snapshots_dmd))
    μ_true = [weight_stats(snapshots_true[k])[1] for k in 1:nt]
    μ_dmd  = [weight_stats(snapshots_dmd[k])[1] for k in 1:nt]

    p = plot(
        tgrid[1:nt],
        μ_true,
        label="true",
        lw=3,
        xlabel="time",
        ylabel="mean weight",
        title="Mean Pauli weight",
    )
    plot!(p, tgrid[1:nt], μ_dmd, label="DMD", lw=2, ls=:dash)
    return p
end

function plot_reconstruction_comparison(
    tgrid,
    snapshots_true::Vector{<:AbstractVector},
    snapshots_dmd::Vector{<:AbstractVector}
)

    Wtrue = stack_snapshots(snapshots_true)
    Wdmd  = stack_snapshots(snapshots_dmd)

    nt = min(length(tgrid), size(Wtrue, 2), size(Wdmd, 2))
    t = tgrid[1:nt]

    Wtrue = Wtrue[:, 1:nt]
    Wdmd  = Wdmd[:, 1:nt]

    # Shared color scale
    cmin = min(minimum(Wtrue), minimum(Wdmd))
    cmax = max(maximum(Wtrue), maximum(Wdmd))

    p1 = heatmap(
        t,
        0:size(Wtrue,1)-1,
        Wtrue,
        xlabel = "time",
        ylabel = "Pauli weight",
        title = "Original",
        clims = (cmin, cmax),
    )

    p2 = heatmap(
        t,
        0:size(Wdmd,1)-1,
        Wdmd,
        xlabel = "time",
        ylabel = "Pauli weight",
        title = "DMD reconstruction",
        clims = (cmin, cmax),
    )

    return p1, p2
end

function relative_error(X, Y)
    return norm(X - Y) / max(norm(X), eps())
end

# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

function main_poc(; q::Int=3, r::Union{Nothing,Int}=4)
    Random.seed!(1)

    # Physical setup
    N = 6
    Jx = 1.0
    Jy = 1.0
    Jz = 1.0
    gx = 0.0
    gy = 0.0
    gz = 0.0

    # Evolution setup
    n_intervals = 100
    total_time = 10.0
    dt = total_time / n_intervals

    # Initial state/operator
    ket = Ket(N, 1)
    o = PauliSum(Pauli(N, X=[1]))

    # Hamiltonian
    H = heisenberg_1D(N, Jx, Jy, Jz; x=gx, y=gy, z=gz)

    # Generate snapshots
    rRES, iRES, tgrid, snapshots = evolve_snapshots(ket, o, H, n_intervals, dt; thresh=1e-12)

    # Takens embedding + DMD fit
    out = embed_fit_summary(snapshots; q=q, r=r)
    res = out.res

    @printf("\nTakens embedding q = %d\n", q)
    @printf("Embedded matrix size: %d x %d\n", size(out.X, 1), size(out.X, 2))
    @printf("Relative linear fit residual: %.4e\n", out.rel_fit)
    @printf("Rank for 95%% energy: %d\n", out.rank95)
    @printf("Rank for 99%% energy: %d\n", out.rank99)

    print_dmd_summary(res; dt=dt, topk=5)

    # Reconstruction in embedded space
    Xhat = reconstruct_dmd(res, size(out.X, 2))

    # Extract the physical block (first delayed copy)
    d = length(snapshots[1])
    m = length(snapshots)

    snapshots_dmd_phys = unembed_physical_average(Xhat, d, q, m)

    # For fair comparison, use the original snapshots on the same time grid
    snapshots_true_phys = snapshots
    tgrid_phys = tgrid

    # Time grid consistent with embedded data
    tgrid_emb = tgrid[1:size(Xhat, 2)]

    # Error diagnostics in physical space
    Wtrue = stack_snapshots(snapshots_true_phys)
    Wdmd  = stack_snapshots(snapshots_dmd_phys)
    recon_err = relative_error(Wtrue, Wdmd)
    μ_true = mean_weight_series(snapshots_true_phys)
    μ_dmd  = mean_weight_series(snapshots_dmd_phys)
    μ_err = relative_error(μ_true, μ_dmd)

    @printf("Reconstruction error (physical block): %.4e\n", recon_err)
    @printf("Mean-weight error:                   %.4e\n", μ_err)

    # Plots
    p1, p2 = plot_reconstruction_comparison(
    tgrid_phys,
    snapshots_true_phys,
    snapshots_dmd_phys,
    )

    p3 = plot_dmd_singular_values(res)    
    p4 = plot_mean_weight(tgrid_phys, snapshots_true_phys, snapshots_dmd_phys)

    display(p1)
    display(p2)
    display(p3)
    display(p4)

    return (
        out = out,
        res = res,
        Xhat = Xhat,
        snapshots = snapshots,
        snapshots_true_phys = snapshots_true_phys,
        snapshots_dmd_phys = snapshots_dmd_phys,
        tgrid = tgrid,
        tgrid_emb = tgrid_emb,
        recon_err = recon_err,
        mean_weight_err = μ_err,
    )
end


# ------------------------------------------------------------
# Helpers for comparison
# ------------------------------------------------------------

function plot_weight_heatmap(
    snapshots::Vector{<:AbstractVector};
    tgrid=nothing,
    title_str="Pauli Weight Dynamics",
    clims=nothing
)
    W = stack_snapshots(snapshots)

    if tgrid === nothing
        tgrid = 0:size(W, 2)-1
    end

    @assert length(tgrid) == size(W, 2) "tgrid length must match the number of snapshot columns."

    kwargs = clims === nothing ? NamedTuple() : (; clims=clims)

    heatmap(
        tgrid,
        0:size(W, 1)-1,
        W,
        xlabel="time",
        ylabel="Pauli weight",
        title=title_str,
        kwargs...
    )
end

function plot_weight_heatmap(
    snapshots::Vector{<:AbstractVector};
    tgrid=nothing,
    title_str="Pauli Weight Dynamics",
    clims=nothing
)
    W = stack_snapshots(snapshots)

    if tgrid === nothing
        tgrid = 0:size(W, 2)-1
    end

    @assert length(tgrid) == size(W, 2) "tgrid length must match the number of snapshot columns."

    if clims === nothing
        return heatmap(
            tgrid,
            0:size(W, 1)-1,
            W,
            xlabel="time",
            ylabel="Pauli weight",
            title=title_str,
        )
    else
        return heatmap(
            tgrid,
            0:size(W, 1)-1,
            W,
            xlabel="time",
            ylabel="Pauli weight",
            title=title_str,
            clims=clims,
        )
    end
end

function unembed_physical_average(Xemb::AbstractMatrix, d::Int, q::Int, m::Int)
    ncols = size(Xemb, 2)
    @assert ncols == m - q + 1 "Embedded matrix has inconsistent number of columns."

    snapshots = Vector{Vector{Float64}}(undef, m)

    for t in 1:m
        accum = zeros(Float64, d)
        count = 0

        kmin = max(1, t - q + 1)
        kmax = min(t, ncols)

        for k in kmin:kmax
            j = t - k + 1
            block = real.(Xemb[(j - 1) * d + 1 : j * d, k])
            accum .+= block
            count += 1
        end

        snapshots[t] = accum ./ max(count, 1)
    end

    return snapshots
end

# ------------------------------------------------------------
# Comparative q study
# ------------------------------------------------------------

function reconstructions_q_compare(; qs=[1, 10, 50, 80,90], r::Union{Nothing,Int}=4)
    Random.seed!(1)

    # Physical setup
    N = 6
    Jx = 1.0
    Jy = 1.0
    Jz = 1.0
    gx = 0.0
    gy = 0.0
    gz = 0.0

    # Evolution setup
    n_intervals = 100
    total_time = 10.0
    dt = total_time / n_intervals

    # Initial state/operator
    ket = Ket(N, 1)
    o = PauliSum(Pauli(N, X=[1]))

    # Hamiltonian
    H = heisenberg_1D(N, Jx, Jy, Jz; x=gx, y=gy, z=gz)

    # Generate snapshots once
    rRES, iRES, tgrid, snapshots = evolve_snapshots(ket, o, H, n_intervals, dt; thresh=1e-12)

    m = length(snapshots)
    d = length(snapshots[1])

    # Store reconstructions and diagnostics for each q
    results = Dict{Int,NamedTuple}()

    Wtrue = stack_snapshots(snapshots)

    println("\n================ q-comparison =================")
    println("Original snapshots: ", size(Wtrue))

    for q in qs
        @assert q <= m "q = $q exceeds the number of snapshots ($m)."

        out = embed_fit_summary(snapshots; q=q, r=r)
        res = out.res

        println("\n--- q = $q ---")
        @printf("Embedded matrix size: %d x %d\n", size(out.X, 1), size(out.X, 2))
        @printf("Relative linear fit residual: %.4e\n", out.rel_fit)
        @printf("Rank for 95%% energy: %d\n", out.rank95)
        @printf("Rank for 99%% energy: %d\n", out.rank99)

        Xhat = reconstruct_dmd(res, size(out.X, 2))
        snapshots_dmd_phys = unembed_physical_average(Xhat, d, q, m)

        Wdmd = stack_snapshots(snapshots_dmd_phys)
        recon_err = relative_error(Wtrue, Wdmd)
        μ_true = mean_weight_series(snapshots)
        μ_dmd = mean_weight_series(snapshots_dmd_phys)
        μ_err = relative_error(μ_true, μ_dmd)

        @printf("Physical reconstruction error: %.4e\n", recon_err)
        @printf("Mean-weight error:             %.4e\n", μ_err)

        results[q] = (
            out = out,
            res = res,
            Xhat = Xhat,
            snapshots_dmd_phys = snapshots_dmd_phys,
            recon_err = recon_err,
            mean_weight_err = μ_err,
        )
    end

    # Common color scale across original + all reconstructions
    all_mats = Matrix{Float64}[]
    push!(all_mats, Wtrue)
    for q in qs
        push!(all_mats, stack_snapshots(results[q].snapshots_dmd_phys))
    end

    cmin = minimum(minimum(M) for M in all_mats)
    cmax = maximum(maximum(M) for M in all_mats)
    clims = (cmin, cmax)

    # Build comparative panels
    plots = Any[]

    push!(plots, plot_weight_heatmap(snapshots; tgrid=tgrid, title_str="Original", clims=clims))

    for q in qs
        push!(
            plots,
            plot_weight_heatmap(
                results[q].snapshots_dmd_phys;
                tgrid=tgrid,
                title_str="DMD reconstruction, q = $q",
                clims=clims
            )
        )
    end

    blank = plot(
        framestyle=:none,
        axis=false,
        grid=false,
        legend=false,
        bacground_color=:white,
    )

    plt = plot(
        plots[1], plots[2], plots[3], plots[4], plots[5], plots[6];
        layout=@layout([a b ; c d ; e f ; g h]),
        size=(1600, 1700)
    )

    display(plt)

    return 
end


function reconstructions_thresh_compare(;
    q::Int=50,
    thresholds=[1e-4, 1e-3, 1e-2],
    r::Union{Nothing,Int}=4
)
    Random.seed!(1)

    # Physical setup
    N = 6
    Jx = 1.0
    Jy = 1.0
    Jz = 1.0
    gx = 0.0
    gy = 0.0
    gz = 0.0

    # Evolution setup
    n_intervals = 100
    total_time = 10.0
    dt = total_time / n_intervals

    # Initial state/operator
    ket = Ket(N, 1)
    o = PauliSum(Pauli(N, X=[1]))

    # Hamiltonian
    H = heisenberg_1D(N, Jx, Jy, Jz; x=gx, y=gy, z=gz)

    results = Dict{Float64, NamedTuple}()

    println("\n================ THRESHOLD COMPARISON =================")
    println("Fixed q = ", q)

    for τ in thresholds
        println("\n--- threshold = $τ ---")

        # Evolve with this threshold
        rRES, iRES, tgrid, snapshots = evolve_snapshots(ket, o, H, n_intervals, dt; thresh=τ)

        m = length(snapshots)
        d = length(snapshots[1])

        @assert q <= m "q = $q exceeds the number of snapshots ($m)."

        Wtrue = stack_snapshots(snapshots)
        μ_true = mean_weight_series(snapshots)

        # DMD on Takens embedding of this threshold-dependent evolution
        out = embed_fit_summary(snapshots; q=q, r=r)
        res = out.res

        @printf("Embedded matrix size: %d x %d\n", size(out.X, 1), size(out.X, 2))
        @printf("Relative linear fit residual: %.4e\n", out.rel_fit)
        @printf("Rank for 95%% energy: %d\n", out.rank95)
        @printf("Rank for 99%% energy: %d\n", out.rank99)

        Xhat = reconstruct_dmd(res, size(out.X, 2))
        snapshots_dmd_phys = unembed_physical_average(Xhat, d, q, m)

        Wdmd = stack_snapshots(snapshots_dmd_phys)
        recon_err = relative_error(Wtrue, Wdmd)

        μ_dmd = mean_weight_series(snapshots_dmd_phys)
        μ_err = relative_error(μ_true, μ_dmd)

        @printf("Physical reconstruction error: %.4e\n", recon_err)
        @printf("Mean-weight error:             %.4e\n", μ_err)

        results[τ] = (
            tgrid = tgrid,
            snapshots = snapshots,
            out = out,
            res = res,
            Xhat = Xhat,
            snapshots_dmd_phys = snapshots_dmd_phys,
            recon_err = recon_err,
            mean_weight_err = μ_err,
        )
    end

    # Common color scale across all threshold-dependent evolutions and reconstructions
    all_mats = Matrix{Float64}[]
    for τ in thresholds
        push!(all_mats, stack_snapshots(results[τ].snapshots))
        push!(all_mats, stack_snapshots(results[τ].snapshots_dmd_phys))
    end

    cmin = minimum(minimum(M) for M in all_mats)
    cmax = maximum(maximum(M) for M in all_mats)
    clims = (cmin, cmax)

    # Build comparative panels: original and DMD reconstruction for each threshold
    plots = Any[]

    for τ in thresholds
        push!(
            plots,
            plot_weight_heatmap(
                results[τ].snapshots;
                tgrid=results[τ].tgrid,
                title_str="PP, thresh = $(τ)",
                clims=clims
            )
        )
        push!(
            plots,
            plot_weight_heatmap(
                results[τ].snapshots_dmd_phys;
                tgrid=results[τ].tgrid,
                title_str="DMD reconstruction, thresh = $(τ)",
                clims=clims
            )
        )
    end

    # 6 thresholds -> 12 plots, arrange symmetrically as 3 rows x 4 cols
    plt = plot(
        plots...;
        layout = @layout([
            a b 
            c d
            e f
        ]),
        size = (1600, 1700)
    )

    display(plt)

    return results, plt
end

main_poc(q=50, r=nothing);
reconstructions_q_compare(r=nothing);
#reconstructions_thresh_compare(q=50, r=nothing);