###########################################################
#
# Collection of Optimization/Search Algorithms
# Revisied Simplex Algorithm for Linear Programming
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
# *** FOR TEACHING PURPOSE ONLY ***
###########################################################

#..........................................................
# Pivotal Transformation
# Input: tableau matrix (a) and pivotal position (p,q)
# Output: tableau w.r.t new basis
#..........................................................

function PivotalTransform(a, p, q)
    m, n = size(a)
    b = similar(a, Float64)
    b[p,:] = a[p,:] ./ a[p,q]
    for i = 1:p-1
        b[i,:] = a[i,:] .- b[p,:] .* a[i,q] # vector .* scalar
    end
    for i = p+1:m
        b[i,:] = a[i,:] .- b[p,:] .* a[i,q] # vector .* scalar
    end
    return b
end

#...................................
# Simple test cases
#...................................

a = [1 5 1 0 0 40; 
     2 1 0 1 0 20; 
     1 1 0 0 1 12;
     -3 -5 0 0 0 0];
PivotalTransform(a, 1, 2)

function SelectQ(r)
    n = length(r)
    q = 0
    qv= 0
    for j = 1:n
        if r[j] < qv
            q =  j
            qv=r[j]
        end
    end
    return q
end

function SelectP(x0, xq)
    m = length(x0)
    if m != length(xq)
        println("ERROR: vectors x_q and x_0 does not have the same length.")
        return -1
    end
    p = 0
    pv= Inf
    for i = 1:m
        if xq[i] > 0 
            temp =x0[i]/xq[i] 
            if temp < pv
                p = i
                pv= temp
            end
        end
    end
    return p
end

#..........................................................
# Revisied Simplex Algorithm for Linear Programming given an initial tableau 
# Input: tableau matrix (A) for the initial basic feasible solution
#        basis block (B), initial feasible solution (x)
#        and cost vector (c) ... 
# Output: Minimum solution the objective function
# Note: extra input and output arguments are for the 2-phase algorithm
#..........................................................
function Simplex(A, # tableau
                 B, # Basis block in a tableau 
                 x, # b vector in Ax = b,
                 c, # c vector in object function, c'*x
                 baseB, baseD; # index vector for column basis vectors
                 iterations::Int64 = 128)
    m, n = size(A)
    for k = 1:iterations
        cb = c[baseB]
        cd = c[baseD]
        D = A[:, baseD]
        r = cd - ((cb'*B)*D)'
        q = SelectQ(r)
        if q == 0 # Optimal solution is found
            println("Minimum cost = ", cb'*x)
            return B, x, baseB, baseD
        end
        xq = B*D[:, q]  
        p = SelectP(x, xq)
        if p == 0 # Not-bounded
            println("Not bounded!")
            return B, x, baseB, baseD
        end
        B = PivotalTransform(hcat(B, x, xq), p, m+2)
        x = B[:, m+1]
        B = B[:, 1:m]
        println("B = ", B)
        temp = baseB[p]
        baseB[p] = baseD[q] # Basis exchange
        baseD[q] = temp
    end
    println("WARNING: ", iterations ," iterations have been exceeded.")
    return B, x, baseB, baseD
end
# A wrapper of Simplex above, given that an initial feasible solution is known
function Simplex(A, # A matrix in Ax = b, should be a tableau 
                 b, # b vector in Ax = b,
                 c, # c vector in object function, c'*x
                 bases; # index vector for column basis vectors
                 iterations::Int64 = 128)
    m, n = size(A)
    if m != length(bases)
        println("ERROR: # of basis vectors does not match with the input tableau")
        return A, b, bases
    end
    baseB = bases
    baseD = setdiff(1:n, bases)
    Simplex(A, eye(Float64, m), b, c, baseB, baseD, iterations = iterations)
end


#...................................
# Simple test cases
#...................................

a = [1 5 1 0 0; 
     2 1 0 1 0; 
     1 1 0 0 1];
b = [40, 20, 12];
c = [-3, -5, 0, 0, 0];
Simplex(a, b, c, [3, 4, 5])

# Reference Solution:
# Minimum cost = -50.0
# ([0.25 0.0 -0.25; 0.25 1.0 -2.25; -0.25 0.0 1.25], [7.0, 3.0, 5.0], ...)

a = [2 1 1 0;
     1 4 0 1];
b = [3, 4]
c = [-7, -6, 0, 0];
Simplex(a, b, c, [3, 4])

# Reference Solution:
# Minimum cost = -12.285714285714285
# ([0.571429 -0.142857; -0.142857 0.285714], [1.14286, 0.714286], ...)


a = [1 0 1 0 0;
     0 1 0 1 0;
     1 1 0 0 1]
b = [4, 6, 8];
c = [-2, -5, 0, 0, 0];
Simplex(a, b, c, [3, 4, 5])

# Reference Solution:
# Minimum cost = -34.0
# ([1.0 1.0 -1.0; 0.0 1.0 0.0; 0.0 -1.0 1.0], [2.0, 6.0, 2.0], ...)

# Phase 1
a = [1 1 1  0 0;
     5 3 0 -1 1];
b = [4, 8];
c = [0, 0, 0, 0, 1];
Bi, x0, bB, bD = Simplex(a, b, c, [3, 5])

# Reference Solution: 
#Minimum cost = 0.0
# ([1.0 -0.2; 0.0 0.2], [2.4, 1.6], ...)
# Phase 2
bE = [5]; # Extra basis vectors
bD = [2, 4];
bB=[3,1];
c = [-3, -5, 0, 0];
Simplex(a[:, 1:4], Bi, x0, c, bB, bD)

#..........................................................
# Two-Phase Simplex Algorithm for Linear Programming  
# Phase 1: Looking for the initial basic feasible solution,
#           through auxiliary vectors
# Phase 2: Searching for optimal solution 
#..........................................................
     
function SimplexTwo(a, # A matrix in Ax = b 
                    b, # b vector in Ax = b,
                    c; # c vector in object function, c'*x
                    iterations::Int64 = 128
                   )
    m, n = size(a)
    #Phase 1
    Bi, x0, bB, bD=Simplex(hcat(a, eye(Float64, m)), 
                           b, 
                           vcat(zeros(Float64, n), ones(Float64, m)),
                           [i for i=n+1:n+m])
    # Phase 2
    Simplex(a, Bi, x0, c, bB, setdiff(bD, n+1:n+m))
end

#...................................
# Simple test cases
#...................................

a = [4 2 -1  0;
     1 4  0 -1];
b = [12, 6];
c = [2, 3, 0, 0];
SimplexTwo(a, b, c)

# Minimum cost = 7.714285714285714
# ([0.285714 -0.142857; -0.0714286 0.285714], [2.57143, 0.857143], [1, 2], [3, 4])

a = [1 1 1 0 0;
     5 3 0 -1 1];
b = [4, 8];
c = [-3, -5, 0, 0, 0];
SimplexTwo(a, b, c)

