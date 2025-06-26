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
    H = hubbard_model_2D(Lx, Ly, t, U, k, pbc) 

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