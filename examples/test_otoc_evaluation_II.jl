using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
"""
 Here, we explore the OTOC estimation for the 1D XXZ Heisenberg
 model.
"""
function clip_coeff!(p::Dict{PauliBasis{N}, T}; thresh=1e-3) where {N, T}
    filter!(p-> abs(p.second) ≥ thresh, p)
end

"""
 The following funciton computes the OTOC expectation value between two operators
 and an initial state (ket).
 """
function otoc_expval(o_1::PauliSum{N}, o_2::PauliSum{N}, ket) where {N}
    output = 0

    for (oi_1, coeff_1) in o_1
        for (oi_2, coeff_2) in o_2
            output += expectation_value(oi_1*oi_2, ket)*coeff_1*coeff_2
        end
    end

    return output
end

"HS norm in Pauli basis: sum_P |c_P|^2."
function hs_norm2(ps::PauliSum)
    s = 0.0
    @inbounds for (_, c) in ps
        s += abs2(c)
    end
    return s
end

"Post-merge pruning in one place."
function prune_post!(ps::PauliSum; scheme::Symbol=:none, mag_thresh::Float64=0.0, wmax::Union{Nothing,Int}=nothing)
    if scheme == :mag || scheme == :combined
        (mag_thresh > 0) && filter!(kv -> abs(kv.second) ≥ mag_thresh, ps)
    end
    if scheme == :weight || scheme == :combined
        (wmax !== nothing) && UnitaryPruning.clip_pauli_weight!(ps, max_w = wmax)
    end
    return ps
end

"""
Pauli-propagation OTOC time series with pruning controls.

Samples once per physical step (3*(N-1) layers), returns F(t) values (OTOC),
snapshots (if requested), plus complexity diagnostics.

Arguments:
- generators, angles  : from UnitaryPruning.heisenberg_1D(...)
- o                   : PauliSum (used as both V and W by default)
- ket                 : state for estimator (your current approach)
- scheme              : :none | :mag | :weight | :combined
- thresh              : magnitude threshold (|coeff|)
- wmax                : max Pauli string length (Int or nothing)
- collect_snapshots   : store PauliSum at each sampled time

Returns:
(F::Vector{Float64}, dicts::Vector{PauliSum}, n_terms::Vector{Int}, loss::Vector{Float64})
where F is your F(t)-type OTOC, dicts are per-layer snapshots only if requested,
n_terms is term count per sample, loss = 1 - HS_norm2 per sample.
"""
function evolution_otoc(generators::Vector{Pauli{N}}, angles,
                        o::PauliSum{N}, ket;
                        scheme::Symbol=:mag,
                        thresh::Float64=1e-3,
                        wmax::Union{Nothing,Int}=nothing,
                        collect_snapshots::Bool=false) where {N}

    nt = length(angles)
    @assert length(generators) == nt "generators/angles length mismatch"
    vcos = cos.(angles); vsin = sin.(angles)

    step = 3*(N-1)                      # layers per physical time step
    nsamp = Int(nt ÷ step) + 1          # include t=0
    Fvals  = Vector{Float64}(undef, nsamp)
    nterms = Vector{Int}(undef, nsamp)
    loss   = Vector{Float64}(undef, nsamp)

    Wt = deepcopy(o)                    # evolve W ≡ o
    V  = o                              # V ≡ o by default (keeps your old behavior)

    # optional layer-by-layer dict dump (kept for compatibility with your code)
    dicts = collect_snapshots ? Vector{Any}(undef, nt) : Any[]

    # t = 0 sample
    os0 = V * Wt
    F0  = real(otoc_expval(os0, os0, ket))  # F(0)=1 for Pauli V=W on any ket
    Fvals[1]  = F0
    nterms[1] = length(Wt)
    n2 = hs_norm2(Wt); loss[1] = 1 - n2

    sample_idx = 2

    for t in 1:nt
        g  = generators[t]
        pb = PauliBasis(g)
        next_W = PauliSum(N)

        @inbounds for (oi, ci) in Wt
            if PauliOperators.commute(oi, pb)
                sum!(next_W, oi * ci)
            else
                # cos branch
                sum!(next_W, oi * (ci * vcos[t]))
                # sin branch (optionally pre-prune the PRODUCED Pauli by weight)
                oj = g * oi
                sum!(next_W, oj * (ci * vsin[t] * 1im))
                end
            end


        # --- POST pruning on the MERGED sum (recommended default) ---
        prune_post!(next_W; scheme=scheme, mag_thresh=thresh, wmax=wmax)
        # -----------------------------------------------------------

        Wt = next_W
        if collect_snapshots
            dicts[t] = deepcopy(Wt)
        end

        # sample after each physical step
        if t % step == 0
            os = V * Wt
            F  = real(otoc_expval(os, os, ket))
            Fvals[sample_idx]  = F
            nterms[sample_idx] = length(Wt)
            n2 = hs_norm2(Wt); loss[sample_idx] = 1 - n2
            sample_idx += 1
        end
    end

    return Fvals, dicts, nterms, loss
end

function run(; N=6, threshold=1e-3, dt=0.1, T=10.0, scheme=:mag,
             wmax::Union{Nothing,Int}=nothing, keep_layers=false, saveprefix="test")

    ket = Ket(N, 0)
    o   = Pauli(N, Z=[1])
    k   = Int(floor(T/dt))
    x,y,z = 0.9,0.9,0.5

    gens, params = UnitaryPruning.heisenberg_1D(o, Jx=x*dt, Jy=y*dt, Jz=z*dt, k=k)
    F, dict, nterms, loss = evolution_otoc(gens, params, PauliSum(o), ket;
                                           scheme=scheme, thresh=threshold, wmax=wmax,
                                           collect_snapshots=keep_layers)

    # time grid with t=0 included
    tgrid = collect(0:dt:T)

    # main OTOC plot
    plt1 = plot(tgrid, F, lw=2, label="$(scheme), th=$(threshold), w=$(wmax === nothing ? "All" : string(wmax))")
    xlabel!(plt1, "Time"); ylabel!(plt1, "OTOC(t)")
    title!(plt1, "N=$N; Jx=$x Jy=$y Jz=$z; dt=$dt")

    # complexity & loss
    plt2 = plot(tgrid, nterms, lw=2, label="n_terms")
    xlabel!(plt2, "Time"); ylabel!(plt2, "# Pauli terms"); title!(plt2, "Complexity growth")
    plt3 = plot(tgrid, loss, lw=2, label="1 - HS_norm2")
    xlabel!(plt3, "Time"); ylabel!(plt3, "lost HS weight"); title!(plt3, "Truncation loss")

    fname_w = (wmax === nothing) ? "All" : string(wmax)
    savefig(plt1, "$(saveprefix)_F_N=$(N)_$(scheme)_th=$(threshold)_w=$(fname_w).pdf")
    savefig(plt2, "$(saveprefix)_nterms_N=$(N)_$(scheme)_th=$(threshold)_w=$(fname_w).pdf")
    savefig(plt3, "$(saveprefix)_loss_N=$(N)_$(scheme)_th=$(threshold)_w=$(fname_w).pdf")

    return (t=tgrid, F=F, n=nterms, loss=loss, dict=dict)
end

#
# - - - Threshold Based Pruning Comparisons - - - 
#
thresholds = [1e-1, 1e-2, 1e-3]
curves = []
NQubits = 8
total_time = 40.0
for th in thresholds
    push!(curves, run(N=NQubits, threshold=th, dt=0.1, T=total_time, scheme=:mag, saveprefix="mag_prune"))
end

# overlay F(t) curves in one figure
plt = plot()
for (i,th) in enumerate(thresholds)
    plot!(plt, curves[i].t, curves[i].F, lw=2, label="th=$(th)")
end
xlabel!("Time"); ylabel!("OTOC(t)"); title!("Coeff Magnitude pruning comparison.")
savefig(plt, "N=$(NQubits)_compare_otoc_coeff_thresh_pruning.pdf")

#
# - - - Weight Based Pruning Comparisons - - - 
#

res_w2 = run(N=NQubits, threshold=0.0, dt=0.1, T=total_time, scheme=:weight,  wmax=2, saveprefix="wcap2")
res_w4 = run(N=NQubits, threshold=0.0, dt=0.1, T=total_time, scheme=:weight,  wmax=4, saveprefix="wcap4")
res_w6 = run(N=NQubits, threshold=0.0, dt=0.1, T=total_time, scheme=:weight,  wmax=6, saveprefix="wcap6")
#res_w8 = run(N=NQubits, threshold=0.0, dt=0.1, T=total_time, scheme=:weight,  wmax=8, saveprefix="wcap8")
#res_w10 = run(N=NQubits, threshold=0.0, dt=0.1, T=total_time, scheme=:weight,  wmax=10, saveprefix="wcap10")


plt_w = plot(res_w2.t, res_w2.F, lw=2, label="wmax=2")
plot!(plt_w, res_w4.t, res_w4.F, lw=2, label="wmax=4")
plot!(plt_w, res_w6.t, res_w6.F, lw=2, label="wmax=6")
#plot!(plt_w, res_w8.t, res_w8.F, lw=2, label="wmax=8")
#plot!(plt_w, res_w10.t, res_w10.F, lw=2, label="wmax=10")
xlabel!("Time"); ylabel!("OTOC(t)"); title!("Weight Pruning comparison.")
savefig(plt_w, "N=$(NQubits)_compare_otoc_weight_pruning.pdf")

#
# - - - Threshold+Weight Based Pruning Comparisons - - - 
#

res_c_1_2  = run(N=NQubits, threshold=1e-1, dt=0.1, T=total_time, scheme=:combined, wmax=2, saveprefix="combined")
res_c_2_2  = run(N=NQubits, threshold=1e-2, dt=0.1, T=total_time, scheme=:combined, wmax=2, saveprefix="combined")
res_c_3_2  = run(N=NQubits, threshold=1e-3, dt=0.1, T=total_time, scheme=:combined, wmax=2, saveprefix="combined")

res_c_1_4  = run(N=NQubits, threshold=1e-1, dt=0.1, T=total_time, scheme=:combined, wmax=4, saveprefix="combined")
res_c_2_4  = run(N=NQubits, threshold=1e-2, dt=0.1, T=total_time, scheme=:combined, wmax=4, saveprefix="combined")
res_c_3_4  = run(N=NQubits, threshold=1e-3, dt=0.1, T=total_time, scheme=:combined, wmax=4, saveprefix="combined")

res_c_1_6  = run(N=NQubits, threshold=1e-1, dt=0.1, T=total_time, scheme=:combined, wmax=6, saveprefix="combined")
res_c_2_6  = run(N=NQubits, threshold=1e-2, dt=0.1, T=total_time, scheme=:combined, wmax=6, saveprefix="combined")
res_c_3_6  = run(N=NQubits, threshold=1e-3, dt=0.1, T=total_time, scheme=:combined, wmax=6, saveprefix="combined")


plt_c = plot(res_c_1_2.t,  res_c_1_2.F,  lw=2, label="combined th=1e-1, w=2",ls=:solid, linecolor=:black)
plot!(plt_c, res_c_2_2.t,  res_c_2_2.F,  lw=2, label="combined th=1e-2, w=2")
plot!(plt_c, res_c_3_2.t,  res_c_3_2.F,  lw=2, label="combined th=1e-3, w=2")
plot!(plt_c, res_c_1_4.t,  res_c_1_4.F,  lw=2, label="combined th=1e-1, w=4")
plot!(plt_c, res_c_2_4.t,  res_c_2_4.F,  lw=2, label="combined th=1e-2, w=4")
plot!(plt_c, res_c_3_4.t,  res_c_3_4.F,  lw=2, label="combined th=1e-3, w=4")
plot!(plt_c, res_c_1_6.t,  res_c_1_6.F,  lw=2, label="combined th=1e-1, w=6")
plot!(plt_c, res_c_2_6.t,  res_c_2_6.F,  lw=2, label="combined th=1e-2, w=6")
plot!(plt_c, res_c_3_6.t,  res_c_3_6.F,  lw=2, label="combined th=1e-3, w=6")   

xlabel!("Time"); ylabel!("OTOC(t)"); title!("Combined pruning comparison.")
savefig(plt_c, "N=$(NQubits)_compare_otoc_combined_pruning.pdf")