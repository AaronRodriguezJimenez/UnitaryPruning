using LinearAlgebra
using SparseArrays

function get_unitary_sequence_1D(o::Pauli{N}; α=.01, k=10) where N
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])
    # print("alpha", α, "\n")
    # Loop over trotter steps
    for ki in 1:k
        ## ZZ layer
        # e^{i π/2 P2} e^{i π P1 /2}|ψ>
        for i in 1:N-1
            pi = Pauli(N, Z=[i, i + 1])
            push!(generators, pi)
            push!(parameters, π/2)
        end
        #pbc 
        pi = Pauli(N, Z=[N, 1])
        push!(generators, pi)
        push!(parameters, π/2)

        ## X layer
        # e^{i αn (-X) / 2}
        for i in 1:N
            pi = Pauli(N, X=[i])
            pi = Pauli{N}((pi.θ + 2)%4, pi.pauli) # this accounts for the fact that the papers have -X and positive ZZ
            push!(generators, pi)
            push!(parameters, α)
        end
    end

    return generators, parameters
end


function get_unitary_sequence_2D(o::Pauli{N}; α=.01, k=10) where N


    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])
    bridges = [[1,5,7], [3,6,9]]
    sequences = [[1,3], [7,9]]

    for ki in 1:k
        # e^{i π/4 P2} e^{i π P1 /2}|ψ>
        ## ZZ layer
        for qubit in sequences
            for i in qubit[1]:qubit[2]
                pi = Pauli(N, Z=[i, i + 1])
                push!(generators, pi)
                push!(parameters, π/2)
            end
        end
        #bridges
        for link in bridges
            pi = Pauli(N, Z=[link[1], link[2]])
            push!(generators, pi)
            push!(parameters, π/2)
            
            pi = Pauli(N, Z=[link[2], link[3]])
            push!(generators, pi)
            push!(parameters, π/2)
        end
        ## X layer
        # e^{i αn Pn / 2}
        for i in 1:N
            pi = Pauli(N, X=[i])
            pi = Pauli{N}((pi.θ + 2)%4, pi.pauli) # this accounts for the fact that the papers have -X and positive ZZ
            push!(generators, pi)
            push!(parameters, α)
        end
    end
    return generators, parameters
end



function build_time_evolution_matrix(generators::Union{Vector{Pauli{N}}, Vector{PauliBasis{N}}}, angles::Vector) where N
    U = Matrix(Pauli(N))
    nt = length(generators)
    length(angles) == nt || throw(DimensionMismatch)
    for t in 1:nt
        α = angles[t]
        U = cos(α/2) .* U .- 1im*sin(α/2) .* U * Matrix(generators[t])

    end

    return U 
end

function build_time_evolution_matrix_fast!(U::AbstractMatrix{ComplexF64}, W::AbstractMatrix{ComplexF64},
    generators::Vector{Pauli{N}},angles::Vector{<:Real}) where N

    nt = length(generators)
    length(angles) == nt || throw(DimensionMismatch())

    fill!(U, 0.0)
    @inbounds for i in axes(U, 1)
        U[i, i] = 1.0
    end

    for t in 1:nt
        α = angles[t]
        Pmat = Matrix(generators[t])

        # W = U * Pmat (in-place)
        mul!(W, U, Pmat)

        # U = cos(α/2)*U - i*sin(α/2)*W (in-place)
        c, s = cos(α / 2), sin(α / 2)
        @inbounds @simd for i in eachindex(U)
            U[i] = c * U[i] - 1im * s * W[i]
        end
    end
    
    return U
end

function build_time_evolution_matrix_fast(N, generators, angles)
    dim = 2^N
    U = Matrix{ComplexF64}(undef, dim, dim)
    W = similar(U)
    return build_time_evolution_matrix_fast!(U, W, generators, angles)
end

function build_time_evolution_sparse!(
    U::AbstractMatrix{ComplexF64},
    W::AbstractMatrix{ComplexF64},
    generators::Vector{<:Any},  # to allow sparse/dense Pauli
    angles::Vector{<:Real}
)
    nt = length(generators)
    length(angles) == nt || throw(DimensionMismatch())

    fill!(U, 0)
    @inbounds for i in axes(U, 1)
        U[i, i] = 1.0
    end

    for t in 1:nt
        α = angles[t]
        Pmat = sparse(Matrix(generators[t]))  # sparse conversion here

        mul!(W, U, Pmat)  # W = U * Pmat

        c, s = cos(α / 2), sin(α / 2)
        @inbounds @simd for i in eachindex(U)
            U[i] = c * U[i] - 1im * s * W[i]
        end
    end

    return U
end

function build_time_evolution_sparse(N, generators, angles)
    dim = 2^N
    U = Matrix{ComplexF64}(undef, dim, dim)
    W = similar(U)
    return build_time_evolution_sparse!(U, W, generators, angles)
end