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
    for i = 1:iterations
        #g = map(f->f(x...), fd)
        g = fd(x...)
        # expensive step
        #h = inv(fh(x...)) 
        # Julia 0.5 does not support this.
        h = inv(Hermitian(fh(x...)))
        d = - alpha*h*g
        x0=x
        x = x + d
        # Verbose output
        if verbose 
            println(i, "\t", x, "\t", f(x...), "\t", d)
        end
        # Test on convergence
        conditions  = [vecnorm(d), abs(f(x...)-f(x0...)), vecnorm(g)]
        denominator = [max(1, vecnorm(d)), max(1, f(x0...)), 1] 
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i)
            println("Norm of last step size: ", vecnorm(d))
            println("Norm of the last gradient:", vecnorm(g))
            return x, f(x...)
        end
    end
    println("Max. iterations have been exceeded.")
    println("Norm of the last gradient: ", vecnorm(fd(x...)))
    return x, f(x...)
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
NewtonConstant( (x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2)  
                          200*(y-x^2)],
                (x, y)-> [2+800*x^2-400*(y-x^2) -400*x;
                          -400*x  200],
                [0, 0], verbose=true)

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


#-----------------------------------------------------------
# Newton's method with the step size corrected to meet Wolfe condition
#-----------------------------------------------------------
function NewtonBacktrack(f,  # Objective function 
                         fd, # Gradient function  
                         fh, # Hessian matrix, only upper triangular elements matter
                         start, # Start point 
                         alpha0 = 1, # Optional factor to control step size, 1 is the original Newton's method
                         tau = 0.5,  # shrinking factor
                         c = 0.5; # control parameter
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
    for i = 1:iterations
        #g = map(f->f(x...), fd)
        g = fd(x...)
        # expensive step
        #h = inv(fh(x...))
        # Julia 0.5 does not support this
        h = inv(Hermitian(fh(x...)))
        d = - h*g
        x0=x
        # Backtracking line search for optimal step size, alpha
        fx = f(x...)
        alpha = alpha0
        fxa = f((x .+ alpha * d)...)
        m = sum(g .* d)
        t = -c * m
        while fx - fxa < alpha * t
            alpha = alpha * tau
            fxa = f((x .+ alpha * d)...)
        end
        x = x0 + alpha*d
        # Verbose output
        if verbose 
            println(i, "\t", x, "\t", f(x...), "\t", alpha)
        end
        # Test on convergence
        conditions  = [alpha*vecnorm(d), abs(f(x...)-f(x0...)), vecnorm(g)]
        denominator = [max(1, alpha*vecnorm(d)), max(1, f(x0...)), 1] 
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i)
            println("Norm of last step size: ", alpha)
            println("Norm of the last gradient:", vecnorm(g))
            return x, f(x...)
        end
    end
    println("Max. iterations have been exceeded.")
    println("Norm of the last gradient: ", vecnorm(fd(x...)))
    return x, f(x...)
end

#-------------------
# Simple test cases
#------------------

NewtonBacktrack((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
               (x, y, z)-> [4*(x-4)^3 
                            2*(y-3)
                            16*(z+5)^3],
               (x, y, z)-> [12*(x-4)^2 0 0;
                            0 2 0;
                            0 0 48*(z+5)^2], 
               [20, 20, 20], verbose=true)

#Rosenbrock function, minimum point [a, a^2] 
#a=1, b=100
NewtonBacktrack((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2)  
                          200*(y-x^2)],
                (x, y)-> [2+800*x^2-400*(y-x^2) -400*x;
                          -400*x  200],
                [0, 0], verbose=true)

#4-dimension Powell function, minimum point [0, 0..., 0]
# search range x_i belongs to (-4, 5)
NewtonBacktrack((x1, x2, x3, x4) -> (x1+10x2)^2+5(x3-x4)^2+(x2-2x3)^4+10(x1-x4)^4,
                (x1, x2, x3, x4) -> [2(x1+10x2)+40(x1-x4)^3
                                    20(x1+10x2)+4(x2-2x3)^3
                                    10(x3-x4)-8(x2-2x3)^3
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

NewtonBacktrack( (x1, x2) -> x1^4+x1*x2+(x1+x2)^3,
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
# Levenberg-Marquardt's algorithm (damped least-squares (DLS) method)
# to solve the problem that Hessian matrix is not positive
#-----------------------------------------------------------
# u, s, v = svd(h + lambda * eye(float64, size(h, 1)))
# d = -(v * diagm(1 ./ s) * u') * g
function NewtonModified(f,  # Objective function 
                         fd, # Gradient function  
                         fh, # Hessian matrix, only upper triangular elements matter
                         start, # Start point 
                         alpha0 = 1, # Optional factor to control step size, 1 is the original Newton's method
                         tau = 0.5,  # shrinking factor
                         c = 0.5; # control parameter
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
    for i = 1:iterations
        fx = f(x...)
        g = fd(x...)
        # expensive step
        #h = inv(fh(x...))
        #h = inv(Hermitian(fh(x...)))
        h = Hermitian(fh(x...))
        d = - g
        x0 = x
        alpha = alpha0
        # Test on invertible
        if abs(det(h)) > eps() # ==0
            if PositiveDefiniteQ(h) 
                d = - inv(h)*g
                alpha = 1
            else
                d = inv(h)*g
                temp = g'*d
                if abs(temp) < eps() # ==0
                    d = -g
                elseif temp > 0
                    d = -d
                end
                # Backtracking line search for optimal step size, alpha
                fxa = f((x .+ alpha * d)...)
                #m = sum(g .* d)
                m = g' * d
                t = -c * m
                while fx - fxa < alpha * t
                    alpha = alpha * tau
                    fxa = f((x .+ alpha * d)...)
                end
            end
        end
        x = x0 + alpha*d
        # Verbose output
        if verbose 
            println(i, "\t", x, "\t", f(x...), "\t", alpha)
        end
        # Test on convergence
        conditions  = [alpha*vecnorm(d), abs(f(x...)-f(x0...)), vecnorm(g)]
        denominator = [max(1, alpha*vecnorm(d)), max(1, f(x0...)), 1] 
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i)
            println("Norm of last step size: ", alpha)
            println("Norm of the last gradient:", vecnorm(g))
            return x, f(x...)
        end
    end
    println("Max. iterations have been exceeded.")
    println("Norm of the last gradient: ", vecnorm(fd(x...)))
    return x, f(x...)
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

NewtonModified((x1, x2) -> x1^4+x1*x2+(1+x2)^2,
                 (x1, x2) -> [4*x1^3+x2
                              x1+2(1+x2)],
                 (x1, x2) -> [12*x1^2 1;
                              1  2],
                 [0, 0], verbose=true)

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

