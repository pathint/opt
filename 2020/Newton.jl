###########################################################
#
# Collection of Optimization/Search Algorithms
# Newton's Algorithms
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
# *** FOR TEACHING PURPOSE ONLY ***
###########################################################

#..........................................................
# Newton's Algorithm, the original version
# Input: function f, 1st derivative f1, Hessian matrix
#        start points m0
# Output: minimmum point and value
#..........................................................

# WARNING: changed convention on input format for fd (and fh)
# Old: fd is a vector of scalar functions, and fh is a matrix of scalar functions 
# Now: fd is a vector-valued function and fh is a matrix-valued function

using LinearAlgebra
function NewtonConstant(f,  # Objective function 
                        fd, # Gradient function  
                        fh, # Hessian matrix, only upper triangular elements matter
                        start, # Start point 
                        alpha = 1;  # Optional factor to contol step size, 1 is the original Newton's method
                        accuracy  ::Float64 = 0.001,
                        accuracy_x::Float64 = accuracy, # arguments | geometry
                        accuracy_f::Float64 = accuracy, # objective | energy
                        accuracy_g::Float64 = accuracy, # gradient  | force
                        convergence_rule = prod, # `prod` means `and (&&)`, `sum` means `or (||)` 
                        convergence_abs = true, # absolute convergence threshold or relative threshold
                        iterations::Int64 = 128, 
                        verbose::Bool = false)
    # Initial setup
    x = start
    pts = []
    push!(pts, x)
    for i = 1:iterations
        #g = map(f->f(x...), fd)
        g = fd(x...)
        # expensive step
        #h = inv(fh(x...)) 
        # Julia 0.5 does not support this.
        #h = inv(Hermitian(fh(x...)))
        h = inv(fh(x...))
        d = - alpha*h*g
        x0=x
        x = x + d
        push!(pts, x)
        # Verbose output
        if verbose 
            println(i, "\t", x, "\t", f(x...), "\t", d)
        end
        # Test on convergence
        conditions  = [norm(d), abs(f(x...)-f(x0...)), norm(g)]
        denominator = [max(1, norm(d)), max(1, f(x0...)), 1] 
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i)
            println("Norm of last step size: ", norm(d))
            println("Norm of the last gradient:", norm(g))
            return x, f(x...), pts
        end
    end
    println("Max. iterations have been exceeded.")
    println("Norm of the last gradient: ", norm(fd(x...)))
    return x, f(x...), pts
end


#...................................
# Simple test cases
#...................................

NewtonConstant((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
               (x, y, z)-> [4*(x-4)^3 
                            2*(y-3)
                            16*(z+5)^3],
               (x, y, z)-> [12*(x-4)^2 0 0;
                            0 2 0;
                            0 0 48*(z+5)^2], 
               [5, 2, -1], verbose=true)

#Rosenbrock function, minimum point [a, a^2] 
#a=1, b=100
# test on start points: [0.0], [100, 100]
x,f,pts=
NewtonConstant( (x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2)  
                          200*(y-x^2)],
                (x, y)-> [2+800*x^2-400*(y-x^2) -400*x;
                          -400*x  200],
                [0, 0], verbose=false)

using DelimitedFiles


#4-dimension Powell function, minimum point [0, 0..., 0]
# search range x_i belongs to (-4, 5)
NewtonConstant( (x1, x2, x3, x4) -> (x1+10x2)^2+5(x3-x4)^2+(x2-2x3)^4+10(x1-x4)^4,
                (x1, x2, x3, x4) -> [2(x1+10x2)+40(x1-x4)^3
                                    20(x1+10x2)+4(x2-2x3)^3
                                    10(x3-x4)-8(x2-2x3)^3
                                    -10(x3-x4)-40(x1-x4)^3],
                (x1, x2, x3, x4) -> [2+120(x1-x4)^2 20 0 -120(x1-x4)^2;
                                     20 200+12(x2-2x3)^2 -24(x2-2x3)^2 0;
                                     0 -24(x2-2x3)^2 10+48(x2-2x3)^2 -10;
                                     -120(x1-x4)^2 0 -10 10+120(x1-x4)^2],
                [3, -1, 0, 1], verbose=true)

# The following is equivalent
NewtonConstant( (x1, x2, x3, x4) -> (x1+10x2)^2+5(x3-x4)^2+(x2-2x3)^4+10(x1-x4)^4,
                (x1, x2, x3, x4) -> [2(x1+10x2)+40(x1-x4)^3
                                    20(x1+10x2)+4(x2-2x3)^3
                                    10(x3-x4)-8(x2-2x3)^3
                                    -10(x3-x4)-40(x1-x4)^3],
                (x1, x2, x3, x4) -> [2+120(x1-x4)^2 20 0 -120(x1-x4)^2;
                                     0 200+12(x2-2x3)^2 -24(x2-2x3)^2 0;
                                     0 0 10+48(x2-2x3)^2 -10;
                                     0 0 0 10+120(x1-x4)^2],
                [3, -1, 0, 1], verbose=true)


# Vector Norms

using LinearAlgebra
#Inexact method, Wolfe condtions
# ϕ(α) <=ϕ(0) + ϵα ϕ'(0)
# ϕ'(0) = d'*g
# ϕ'(α) >= η ϕ'(0)
function search_alpha(f, g, xk, fk, gk, d; α0=1, ϵ=0.1, τ=0.5, η=0.5, ζ=2.0)
    α = α0
    ϕ0= d'*gk
    δ = α .* d
    xn = xk + δ
    fn = f(xn...)
    gn = g(xn...)
    # Armijo condition 
    while fn > fk + ϵ*α*ϕ0
        α = τ*α
        δ = α .* d
        xn = xk + δ
        fn = f(xn...)
        gn = g(xn...)
    end
    # Wolfe condition 
    while d'*gn < η*ϕ0
        α = ζ*α
        δ = α .* d
        xn = xk + δ
        fn = f(xn...)
        gn = g(xn...)
    end
    return α, δ, xn, fn, gn
end

#-----------------------------------------------------------
# Newton's method with the step size corrected to meet Wolfe condition
#-----------------------------------------------------------
function NewtonBacktrack(f,  # Objective function 
                         g,  # Gradient function  
                         h,  # Hessian matrix, only upper triangular elements matter
                         x0;  # Start point 
                         α0 = 1,  # Optional factor to control step size, 1 is the original Newton's method
                         τ = 0.5, # shrinking factor
                         η = 0.5, # border parameter
                         ϵ = 0.1, # border parameter
                         accuracy::Float64   = 0.001,
                         accuracy_x::Float64 = accuracy, # arguments | geometry
                         accuracy_f::Float64 = accuracy, # objective | energy
                         accuracy_g::Float64 = accuracy, # gradient  | force
                         convergence_rule = prod, # `prod` means `and (&&)`, `sum` means `or (||)` 
                         convergence_abs = true, # absolute convergence threshold or relative threshold
                         iterations::Int64 = 128, 
                         verbose::Bool = false)
    # Initial setup
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    pts = []
    push!(pts, xk)
    for i = 1:iterations
        # Inverse of the hessian matrix, very expensive step
        hk = inv(h(xk...))
        d = - hk*gk
        # Search directions
        # Backtracking line search for optimal step size, alpha
        α, δ, xn, fn, gn  = search_alpha(f, g, xk, fk, gk, d,
                                         α0 = α0, ϵ = ϵ, τ = τ, η = η, ζ = 1/τ)
        # Verbose output
        if verbose 
            println("i:", i, "\tx:", xk, "\tf(x):", fk, "\tα:", α)
        end
        # Test on convergence
        conditions  = [norm(δ), abs(fn - fk), norm(gn)]
        denominator = [max(1, norm(δ)), max(1, fk), 1] 
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps:\t", i)
            println("Norm of last step size:\t", α)
            println("Norm of the last gradient:\t", norm(gn))
            return xn, fn, pts
        end
        push!(pts, xn)
        xk = xn
        fk = fn
        gk = gn
    end
    println("Max. iterations have been exceeded.")
    println("Norm of the last gradient: ", norm(gk))
    return xk, fk, pts
end

#-------------------
# Simple test cases
#------------------

NewtonBacktrack((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
               (x, y, z)-> [4*(x-4)^3, 
                            2*(y-3),
                            16*(z+5)^3],
               (x, y, z)-> [12*(x-4)^2 0 0;
                            0 2 0;
                            0 0 48*(z+5)^2], 
               [20, 20, 20], verbose=true)

#Rosenbrock function, minimum point [a, a^2] 
#a=1, b=100
x,f,pts=
NewtonBacktrack((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2),  
                          200*(y-x^2)],
                (x, y)-> [2+800*x^2-400*(y-x^2) -400*x;
                          -400*x  200],
                [0, 0], verbose=true)

using DelimitedFiles
writedlm("rosenbrock_bk.tsv", pts)

#4-dimension Powell function, minimum point [0, 0..., 0]
# search range x_i belongs to (-4, 5)
NewtonBacktrack((x1, x2, x3, x4) -> (x1+10x2)^2+5(x3-x4)^2+(x2-2x3)^4+10(x1-x4)^4,
                (x1, x2, x3, x4) -> [2(x1+10x2)+40(x1-x4)^3,
                                    20(x1+10x2)+4(x2-2x3)^3,
                                    10(x3-x4)-8(x2-2x3)^3,
                                    -10(x3-x4)-40(x1-x4)^3],
                (x1, x2, x3, x4) -> [2+120(x1-x4)^2 20 0 -120(x1-x4)^2;
                                     20 200+12(x2-2x3)^2 -24(x2-2x3)^2 0;
                                     0 -24(x2-2x3)^2 10+48(x2-2x3)^2 -10;
                                     -120(x1-x4)^2 0 -10 10+120(x1-x4)^2],
                [10, -5, 15, 5], verbose=true)

# Not positive definite Hessian matrix
NewtonConstant( (x1, x2) -> x1^4+x1*x2+(x1+x2)^3,
                 (x1, x2) -> [4*x1^3+x2
                              x1+2(1-x2)],
                 (x1, x2) -> [12*x1^2 1;
                              1  2],
                 [0, 0])

#-----------------------------------------------------------
# Test if a matrix is positive-definite nor not 
# Sylvester's criterion
#------------------------------------------------------------
function PositiveDefiniteQ(A)
    if ndims(A) != 2
        return false # WARNING: should raise an error
    end
    n, m = size(A)
    if n != m 
        return false # WARNING: should raise an error
    end
    if n < 1  # Redundant?
        return false # WARNING: should raise an error
    end
    for i = 1:n
        if det(A[1:i, 1:i]) < 0
            return false
        end
    end
    return true
end

#-----------------------------------------------------------
# Modified Newton's algorithm,
# handling the singular and non-positive definite Hessian matrices
#-----------------------------------------------------------
# u, s, v = svd(h + lambda * eye(float64, size(h, 1)))
# d = -(v * diagm(1 ./ s) * u') * g
function NewtonModified(f,  # Objective function 
                        g, # Gradient function  
                        h, # Hessian matrix, only upper triangular elements matter
                        x0;  # Start point 
                        α0 = 1,  # Optional factor to control step size, 1 is the original Newton's method
                        τ = 0.5, # shrinking factor
                        η = 0.5, # border parameter
                        ϵ = 0.1, # border parameter
                        accuracy  ::Float64 = 0.001,
                        accuracy_x::Float64 = accuracy, # arguments | geometry
                        accuracy_f::Float64 = accuracy, # objective | energy
                        accuracy_g::Float64 = accuracy, # gradient  | force
                        convergence_rule = prod, # `prod` means `and (&&)`, `sum` means `or (||)` 
                        convergence_abs = true, # absolute convergence threshold or relative threshold
                        iterations::Int64 = 128, 
                        verbose::Bool = false)
    # Initial setup
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    pts = []
    push!(pts, xk)
    for i = 1:iterations
        # Inverse of the hessian matrix, very expensive step
        hk = h(xk...)
        # Another expensive step
        em = eigmin(hk)
        if em < 0
           hk = hk - 1.1*em * I 
        end
        # Yet another expensive step
        hk = inv(hk)
        d = - hk*gk
        # Search direction
        # Backtracking line search for optimal step size, alpha
        α, δ, xn, fn, gn  = search_alpha(f, g, xk, fk, gk, d,
                                         α0 = α0, ϵ = ϵ, τ = τ, η = η, ζ = 1/τ)
        # Verbose output
        if verbose 
            println("i:", i, "\tx:", xk, "\tf(x):", fk, "\tα:", α)
        end
        # Test on convergence
        conditions  = [norm(δ), abs(fn - fk), norm(gn)]
        denominator = [max(1, norm(δ)), max(1, fk), 1] 
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps:\t", i)
            println("Norm of last step size:\t", α)
            println("Norm of the last gradient:\t", norm(gn))
            return xn, fn, pts
        end
        push!(pts, xk)
        xk = xn
        fk = fn
        gk = gn
    end
    println("Max. iterations have been exceeded.")
    println("Norm of the last gradient: ", norm(gk))
    return xk, fk, pts
end

#-------------------
# Simple test cases
#------------------
NewtonConstant((x1, x2) -> x1^4+x1*x2+(1+x2)^2,
                 (x1, x2) -> [4*x1^3+x2
                              x1+2(1+x2)],
                 (x1, x2) -> [12*x1^2 1;
                              1  2],
                 [0, 0], verbose=true)

NewtonBacktrack((x1, x2) -> x1^4+x1*x2+(1+x2)^2,
                 (x1, x2) -> [4*x1^3+x2
                              x1+2(1+x2)],
                 (x1, x2) -> [12*x1^2 1;
                              1  2],
                 [0, 0], verbose=true)

x,f,pts=
NewtonModified((x1, x2) -> x1^4+x1*x2+(1+x2)^2,
                 (x1, x2) -> [4*x1^3+x2,
                              x1+2(1+x2)],
                 (x1, x2) -> [12*x1^2 1;
                              1  2],
                 [0, 0], verbose=true)

using DelimitedFiles
writedlm("f3_lm.tsv", pts)

NewtonModified((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
               (x, y, z)-> [4*(x-4)^3 
                            2*(y-3)
                            16*(z+5)^3],
               (x, y, z)-> [12*(x-4)^2 0 0;
                            0 2 0;
                            0 0 48*(z+5)^2], 
               [20, 20, 20], verbose=true)

#Rosenbrock function, minimum point [a, a^2] 
#a=1, b=100
NewtonModified((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2)  
                          200*(y-x^2)],
                (x, y)-> [2+800*x^2-400*(y-x^2) -400*x;
                          -400*x  200],
                [100, 100], verbose=true)

#4-dimension Powell function, minimum point [0, 0..., 0]
# search range x_i belongs to (-4, 5)
NewtonModified((x1, x2, x3, x4) -> (x1+10x2)^2+5(x3-x4)^2+(x2-2x3)^4+10(x1-x4)^4,
                (x1, x2, x3, x4) -> [2(x1+10x2)+40(x1-x4)^3
                                    20(x1+10x2)+4(x2-2x3)^3
                                    10(x3-x4)-8(x2-2x3)^3
                                    -10(x3-x4)-40(x1-x4)^3],
                (x1, x2, x3, x4) -> [2+120(x1-x4)^2 20 0 -120(x1-x4)^2;
                                     20 200+12(x2-2x3)^2 -24(x2-2x3)^2 0;
                                     0 -24(x2-2x3)^2 10+48(x2-2x3)^2 -10;
                                     -120(x1-x4)^2 0 -10 10+120(x1-x4)^2],
                [10, -1, 5, 5], verbose=true)

using LinearAlgebra
#-----------------------------------------------------------
# Least-squares nonlinear fit Gauss-Newton algorithm
#-----------------------------------------------------------
function GaussNewtonFit(f,  # Model, fitting function, variables followed by model parameters 
                        g, # Gradient (1st derivative) of the fitting function w.r.t fitting parameters   
                        data, # data to be fitted, [x11, x12, x13 ... y1; ...; xm1, xm2, xm3 ... ym ]
                        start; # Guessed parameters 
                        α0 = 1, # Optional factor to control step size, 1 is the original Newton's method
                        accuracy  ::Float64 = 0.001,
                        iterations::Int64 = 128, 
                        verbose::Bool = false)
    # check data size
    if ndims(data) != 2
	error("data should be 2-dimensional array")
    end
    nr, nc = size(data)
    if nc < 2 
	error("data should contain more than 1 column")
    end
    nn = length(start)
    # return a vector
    function residuals(para)
        return mapslices(row->f(row[1:end-1]..., para...)-row[end], data, dims = [2])
    end
    # Initial setup
    x0 = start
    r0 = residuals(x0)
    s0 = sum(r0.^2)
    for i = 1:iterations
        #  m*n matrix
        jacobian = mapslices(row->g(row[1:end-1]..., x0...), data, dims = [2])
        g0 = jacobian' * r0
        h0 = jacobian' * jacobian
        em = eigmin(h0)
        if em < 0
           h0 = h0 - 1.1*em * I 
        end
        d = - inv(h0)*g0
        #Search step size α
        δ  = α0 .* d
        x = x0 .+ δ 
        r = residuals(x)
        s = sum(r.^2)
        # Verbose output
        if verbose 
            println("i:", i, "\tx:", x, "\ts^2:", s)
        end
        # Test on convergence
        if abs(s-s0)/max(1, s0) < accuracy
            println("Number of steps:\t", i)
            println("Norm of the last step size:\t", norm(δ))
            println("Norm of the last gradient:\t",  norm(g0))
            println("Residuals: ", r)
            return x, s
        end
        x0 = x
        r0 = r # No need to store
        s0 = s
    end
    println("Max. iterations have been exceeded.")
    return x0, s0
end

# Best Fit: 0.800399 + 0.0933134^x2
GaussNewtonFit((x, a, b) -> a + b * x^2,
               (x, a, b) -> [1,
                             x^2],
               [0 1;
		1 0;
		3 2;
		5 4;
		6 4;
		7 5],
               [1, 1], verbose=true)
               
GaussNewtonFit((x, a, b) -> a + b * x^2,
               (x, a, b) -> [1
                             x^2],
               [0 1;1 0;3 2;5 4;6 4;7 5],
               [0, 0], verbose=true)

# Best Fit: 0.2868 0.00336784
GaussNewtonFit((x, a, b) -> a + exp(b*x^2),
               (x, a, b) -> [1,
                             exp(b*x^2)*x^2],
               [0 1;1 0;3 2;5 4;6 4;7 5],
               [0, 0], verbose=true)

GaussNewtonFit((x, a, b) -> a + exp(b*x^2),
               (x, a, b) -> [1
                             exp(b*x^2)*x^2],
               [0 1;1 0;3 2;5 4;6 4;7 5],
               [1., 1.], verbose=true)

