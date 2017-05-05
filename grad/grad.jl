###########################################################
#
# Collection of Optimization/Search Algorithms
# Gradient Descent Algorithms
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
# *** FOR TEACHING PURPOSE ONLY ***
###########################################################

#..........................................................
# Gradient Descent (Steepest Descent) Algorithm
# Input: function f and its 1st derivative f1
#        start points m0, accuracy and optional max. iterations
# Output: minimmum point and value
#..........................................................

function GradientConstant(f,  # Objective function 
                          fd, # Gradient functions  
                          start, # Start point 
                         alpha;  # Step size
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
        g = map(f->f(x...), fd)
        #d = - alpha*g
        d = - alpha*normalize(g)
        x0=x
        x = x + d
        # Verbose output
        if verbose 
            println(i, "\t", x, "\t", f(x...))
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
    println("Norm of the last gradient: ", vecnorm(map(f->f(x...), fd)))
    return x, f(x...)
end

#...................................
# Simple test cases
#...................................

GradientConstant((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
                [(x, y, z)-> 4*(x-4)^3,
                 (x, y, z)-> 2*(y-3),
                 (x, y, z)-> 16*(z+5)^3],
                [5, 2, -1], 1)

GradientConstant((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
                [(x, y, z)-> 4*(x-4)^3,
                 (x, y, z)-> 2*(y-3),
                 (x, y, z)-> 16*(z+5)^3],
                [5, 2, -1], 1, verbose = true)

GradientConstant((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
                [(x, y, z)-> 4*(x-4)^3,
                 (x, y, z)-> 2*(y-3),
                 (x, y, z)-> 16*(z+5)^3],
                [5, 2, -1], 0.001, iterations = 10^6)

#-----------------------------------------------------------
# Steepest descent with binary search for step size
#-----------------------------------------------------------
function GradientBinary( f,  # Objective function 
                         fd, # Gradient functions  
                         start, # Start point 
                         alpha0 = 1; # Initial alpha value
                         accuracy  ::Float64 = 0.001,
                         accuracy_1::Float64 = accuracy, # accuracy for step size optimzation
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
        g = map(f->f(x...), fd)
        # Search direction
        d = - normalize(g)
        x0=x
        # 1-D binary search for optimal step size, alpha
        iter1d = ceil(log2(alpha0/abs(accuracy_1)))
        u = alpha0
        l = 0
        m = (l + u)/2
        for j = 1:iter1d
            f1m = -sum(g .* map(f->f((x .+ m*d )...), fd))
            #println("Iter ", j, "\t D ", f1m)
            if abs(f1m) < accuracy_1 
               alpha = m
               break
            elseif f1m >0
               u = m
            else
               l = m
            end
            m = (l + u)/2
        end
        alpha = m   
        x = x0 + alpha*d
        # Verbose output
        if verbose 
            println(i, "\t", x, "\t", f(x...), "\t", alpha)
        end
        # Test on convergence
        conditions  = [alpha, abs(f(x...)-f(x0...)), vecnorm(g)]
        denominator = [max(1, alpha), max(1, f(x0...)), 1] 
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
    println("Norm of the last gradient: ", vecnorm(map(f->f(x...), fd)))
    return x, f(x...)
end

#-------------------
# Simple test cases
#------------------

GradientBinary((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
                [(x, y, z)-> 4*(x-4)^3,
                 (x, y, z)-> 2*(y-3),
                 (x, y, z)-> 16*(z+5)^3],
                [5, 2, -1])

GradientBinary((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
                [(x, y, z)-> 4*(x-4)^3,
                 (x, y, z)-> 2*(y-3),
                 (x, y, z)-> 16*(z+5)^3],
                [0, 0, 0], verbose=true, convergence_abs=false)

GradientBinary( (x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                  [(x, y)-> -2*(1-x)-400*x*(y-x^2),
                   (x, y)-> 200*(y-x^2)],
                  [-1, 1], verbose=true)
                  
GradientBinary( (x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                  [(x, y)-> -2*(1-x)-400*x*(y-x^2),
                   (x, y)-> 200*(y-x^2)],
                  [-3, -4], iterations=10^4)

#-----------------------------------------------------------
# Steepest descent with backtracking line search for step size
#-----------------------------------------------------------
function GradientBacktrack(f,  # Objective function 
                         fd, # Gradient functions  
                         start, # Start point 
                         alpha0 = 1, # Initial alpha value
                         tau = 0.5, # shrinking factor
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
        g = map(f->f(x...), fd)
        # Search direction
        d = - normalize(g)
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
        conditions  = [alpha, abs(f(x...)-f(x0...)), vecnorm(g)]
        denominator = [max(1, alpha), max(1, f(x0...)), 1] 
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
    println("Norm of the last gradient: ", vecnorm(map(f->f(x...), fd)))
    return x, f(x...)
end

#-------------------
# Simple test cases
#------------------

GradientBacktrack((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
                [(x, y, z)-> 4*(x-4)^3,
                 (x, y, z)-> 2*(y-3),
                 (x, y, z)-> 16*(z+5)^3],
                [5, 2, -1], verbose=true)

GradientBacktrack((x, y, z)-> (x-4)^4 + (y-3)^2 + 4*(z+5)^4,
                [(x, y, z)-> 4*(x-4)^3,
                 (x, y, z)-> 2*(y-3),
                 (x, y, z)-> 16*(z+5)^3],
                [0, 0, 0])

GradientBacktrack( (x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                  [(x, y)-> -2*(1-x)-400*x*(y-x^2),
                   (x, y)-> 200*(y-x^2)],
                  [-1, 1], iterations = 10^4)
                  
GradientBacktrack( (x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                  [(x, y)-> -2*(1-x)-400*x*(y-x^2),
                   (x, y)-> 200*(y-x^2)],
                  [-3, -4], iterations = 10^4)

