function JWmapping(a::Int, b::Int, N::Int)
    # Compute C^dagger_i term
    ax_term = Pauli(2^(a-1)-1, 2^(a-1), N)
    ay_term = Pauli(2^(a)-1, 2^(a-1), N)
    c_dagg_a = 0.5 * (ax_term - ay_term)

    # Compute C_j term
    bx_term = Pauli(2^(b-1)-1, 2^(b-1), N)
    by_term = Pauli(2^(b)-1, 2^(b-1), N)
    c_b = 0.5 * (bx_term + by_term)

    # Build C^dagger_i*C_j
    term =  c_dagg_a*c_b

    return term
end

"""
 The following function performs the Jordan-Wirgner mapping for fermionic 
    bilinear terms 
    N - Total number of fermionic modes
    a,b, - indices of the modes to be mapped
    returns term = c^dagger_a * c_b
"""
function JWmapping_ver_two(o::Pauli{N}; i::Int, j::Int) where N
    # Compute C^dagger_i term
    ax_term = Pauli(2^(i-1)-1, 2^(i-1), N)
    ay_term = Pauli(2^(i)-1, 2^(i-1), N)
    c_dagg_a = 0.5 * (ax_term - ay_term)

    # Compute C_j term
    bx_term = Pauli(2^(j-1)-1, 2^(j-1), N)
    by_term = Pauli(2^(j)-1, 2^(j-1), N)
    c_b = 0.5 * (bx_term + by_term)

    # Build C^dagger_i*C_j
    term =  c_dagg_a*c_b

    return term
end

"""
 The following version of the Hubbard model in 2D incorporates the option of handling 
 periodic boundary conditions by setting pbc = true, as:
 - The left edge connects to the right edge (in x).
 - The top edge connects to the bottom edge (in y).
 Otherwise it reduces to an open boundary condition model (OBC) for which 
 no connections are made beyond the lattice boundaries.
"""
function hubbard_model_2D(Lx::Int, Ly::Int, t::Float64, U::Float64, k::Int, pbc::Bool)
    D = Lx * Ly           # number of spatial sites
    N = 2 * D             # total number of spin orbitals (up and down)
    H_hop = PauliSum(N)
    H_u = PauliSum(N)

    for kl in 1:k
        # - - - Hopping term - - -
        for y in 0:(Ly - 1)
            for x in 0:(Lx - 1)
                i = y * Lx + x + 1  # site index (1-based)

                # Right neighbor (x-direction)
                if pbc || x < Lx - 1
                    x_nbr = (x + 1) % Lx
                    j = y * Lx + x_nbr + 1
                    for spin in 0:1
                        a = i + spin * D
                        b = j + spin * D
                        H_hop += JWmapping(a,b,N)
                        H_hop += JWmapping(b,a,N)
                    end
                end

                # Bottom neighbor (y-direction)
                if pbc || y < Ly - 1
                    y_nbr = (y + 1) % Ly
                    j = y_nbr * Lx + x + 1
                    for spin in 0:1
                        a = i + spin * D
                        b = j + spin * D
                        H_hop += JWmapping(a,b,N)
                        H_hop += JWmapping(b,a,N)
                    end
                end
            end
        end

        # - - - Interaction term - - -
        for site in 1:D
            a_up = site
            a_dn = site + D
            H_u += JWmapping(a_up, a_up, N) * JWmapping(a_dn, a_dn, N)
        end
    end

    return -t * H_hop + U * H_u
end

function hubbard_model_2D_interleaved(o::Pauli{N}; Lx::Int64, Ly::Int64, t::Float64, U::Float64, k::Int64) where N
    D = Lx * Ly  # number of lattice sites

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    H_hop = PauliSum(N)
    H_u = PauliSum(N)

    # 1-based linear index
    linear_index(x, y) = (y - 1) * Lx + x  # returns 1 to D

    # Spin-orbital index: ↑ = 2j - 1, ↓ = 2j
    up(j) = 2*j - 1
    dn(j) = 2*j

    for kl in 1:k

        H_hop = PauliSum(N)
        H_u = PauliSum(N)

        # Loop over 1-based coordinates
        for x in 1:Lx
            for y in 1:Ly
                i = linear_index(x, y)

                # Right neighbor (x+1)
                if x < Lx
                    j = linear_index(x + 1, y)
                    for (a_fn, b_fn) in [(up, up), (dn, dn)]
                        a = a_fn(i)
                        b = b_fn(j)
                        H_hop += JWmapping_ver_two(o, i=a, j=b)
                        H_hop += JWmapping_ver_two(o, i=b, j=a)
                    end
                end

                # Bottom neighbor (y+1)
                if y < Ly
                    j = linear_index(x, y + 1)
                    for (a_fn, b_fn) in [(up, up), (dn, dn)]
                        a = a_fn(i)
                        b = b_fn(j)
                        H_hop += JWmapping_ver_two(o, i=a, j=b)
                        H_hop += JWmapping_ver_two(o, i=b, j=a)
                    end
                end
            end
        end

        # On-site interaction terms
        for site in 1:D
            a_up = up(site)
            a_dn = dn(site)
            H_u += JWmapping_ver_two(o, i=a_up, j=a_up) * JWmapping_ver_two(o, i=a_dn, j=a_dn)
        end
    end

    return  -t * H_hop + U * H_u
end

# Check eigenspectrum
function hubbard_2d_eigenspectrum()
    Lx = 2
    Ly = 2
    N = 2*Lx*Ly
    k = 1
    o = Pauli(N, Z=[1])
    t = 1.0
    U = 2.0
    pbc = false
    #H = hubbard_model_2D(Lx, Ly, t, U, k, pbc) 
    H = hubbard_model_2D_interleaved(o; Lx=Lx, Ly=Ly, t=t, U=U, k=k)

    e, v = eigen(Matrix(H))
    println("AARON WAS HERE...")
    for eig in e
        println(eig)
    end

    filename = "2D_Hubbard_eigenspectrum_t$t-U$U-Aaron.txt"
    open(filename, "w") do f
        for item in e
            println(f, item)
        end
    end
    return
end

hubbard_2d_eigenspectrum()