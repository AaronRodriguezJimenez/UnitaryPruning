#
#  Here we check the eigenspectrum of the version provided
#  by Chinmay, and based on a different version of PauliOperators
#
function jw_transform(o::Pauli{N}, site) where N
    z_string = [i for i in 1:site-1]
    # p = PauliSum(N)
    p = Pauli(N, Z = z_string, X = [site]) + im * Pauli(N, Z=z_string, Y=[site])
    return 0.5*p
end

function fermi_hubbard_2D(o::Pauli{N}; t, U, k) where N
    Nsites = Int(N/2)
    L = Int(sqrt(Nsites))
    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()
    t_term = PauliSum(N)
    u_term = PauliSum(N)
    up(j) = 2*j - 1
    dn(j) = 2*j
    linear_index(x, y) = (x-1)*L + y
    for ki in 1:k
        t_term = PauliSum(N)
        for x in 1:L
            for y in 1:L
                j = linear_index(x, y)
                if x < L
                    # down coupling
                    i = linear_index(x + 1, y)
                    # α-spin c{i, α}†c{j, α} + h.c.
                    i_a = jw_transform(o, up(j))
                    j_a = jw_transform(o, up(i))
                    t_term += i_a' * j_a
                    t_term += j_a' * i_a
                    # β-spin c{i, β}†c{j, β} + h.c.
                    i_b = jw_transform(o, dn(j))
                    j_b = jw_transform(o, dn(i))
                    t_term += i_b' * j_b
                    t_term += j_b' * i_b
                end
                if y < L
                    # α-spin c{i, α}†c{j, α} + h.c.
                    i = linear_index(x, y + 1)
                    # right side coupling
                    i_a = jw_transform(o, up(j))
                    j_a = jw_transform(o, up(i))
                    t_term += i_a' * j_a
                    t_term += j_a' * i_a
                    # β-spin c{i, β}†c{j, β} + h.c.
                    i_b = jw_transform(o, dn(j))
                    j_b = jw_transform(o, dn(i))
                    t_term += i_b' * j_b
                    t_term += j_b' * i_b
                end
            end
        end
        for (pauli, coeff) in t_term
            push!(generators, Pauli(pauli))
            push!(parameters, -t*coeff)
        end
        u_term = PauliSum(N)
        for j in 1:Nsites
            # interacting term
            i_a = jw_transform(o, up(j))
            i_b = jw_transform(o, dn(j))
            u_term += i_a'*i_a*i_b'*i_b
            # println("Interaction")
            # display(u_term)
        end
        for (pauli, coeff) in u_term
            push!(generators, Pauli(pauli))
            push!(parameters, U*coeff)
        end
    end
    return generators, parameters, -t*t_term + U*u_term
end

# Check eigenspectrum
function hubbard_2d_eigenspectrum()
    N = 2
    N = 2*N*N
    k = 1
    o = Pauli(N, Z=[1])
    t = 1
    U = 4
    generators, parameters, hammy = fermi_hubbard_2D(o, t = t, U = U, k = k)
    e, v = eigen(Matrix(hammy))
    println("CHINMAY WAS HERE...")
    for eig in e
        println(eig)
    end
    filename = "2D_Hubbard_eigenspectrum_t$t-U$U-Chinmay.txt"
    open(filename, "w") do f
        for item in e
            println(f, item)
        end
    end
    return
end

hubbard_2d_eigenspectrum()