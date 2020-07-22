###########################################################
#
# Collection of 1D Optimization/Search Algorithms
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
#
###########################################################

#..........................................................
# Golden Section Algorithm
# Input: function f
#        search range {l0, u0} and accuracy
# Output: minimum point and value
#..........................................................

function GoldenSection(f, l0, u0, accuracy)
    if l0 >= u0
        #println("WARNING: Two bounds have been interchanged!")
        u = l0
        l = u0
    else
        u = u0
        l = l0
    end
    rho = (3 - sqrt(5))/2
    maxIter = ceil(-log(1-rho, (u-l)/abs(accuracy))) 
    # Initial iteration, setup for later iterations
    delta = rho*(u-l)
    u1 = u - delta
    l1 = l + delta
    fu1 = f(u1)
    fl1 = f(l1)
    for i = 2:maxIter
        if fl1 < fu1 # New range, {l, u1}
           u = u1
           u1 = l1
           fu1 = fl1
           l1 = l + rho*(u1-l)
           fl1 = f(l1)
        else # New range, {l1, u}
           l = l1
           l1 = u1
           fl1 = fu1
           u1 = u - rho*(u-l1)
           fu1 = f(u1)
        end
    end
    # Final result
    return (l+u)/2, f((l+u)/2)
end
        


# Golden Section Algorithum for 1-d search
# f: 1-dimension objective function
# a: lower bound of search range
# b: upper bound of search range
function GoldenSection(f, a, b;acc = 0.01, verbose=false)
        # Handle input
        # to ensure the lower
        if a < b
           a0 = a
           b0 = b
         else
           a0 = b
           b0 = a
        end
        # Compression factor
        ρ = (3-sqrt(5))/2
        n = ceil(-log(1-ρ, (b0-a0)/abs(acc))) 
        c = ρ*(b0-a0)
        a1 = a0 + c 
        b1 = b0 - c
        fa1 = f(a1)
        fb1 = f(b1)
        for i=1:n
                # New range: a0..b1
                if fa1 < fb1
                       b0 = b1
                       b1 = a1
                       fb1 = fa1
                       a1 = a0 + ρ*(b0-a0)
                       fa1 = f(a1)
                else # New range: a1..b0
                        a0 = a1
                        a1 = b1
                        fa1 = fb1
                        b1 = b0 - ρ*(b0-a0)
                        fb1 = f(b1)
                end
                if verbose
                     println("i=",i, "\ta0=", a0, "\tb0=", b0)
                end
        end
        if fa1 < fb1
                c=0.5*(a0+b1) 
        else
                c=0.5*(a1+b0)
        end
        return c, f(c)
end

GoldenSection( x -> x^2,
     -1,
      1
   )

GoldenSection( x -> x^2,
     -1,
     1,
     acc=0.00001,
     verbose=true
   )

GoldenSection( x -> x^2 + cos(x) ,
     -1,
     1,
     acc=0.00001,
     verbose=true
   )
#...................................
# Simple test cases
#...................................

GoldenSection(x->x^2, -10, 10, eps())

GoldenSection(x->x^2/2 - sin(x), -20, 20, eps())


#..........................................................
# Fibonacci Section Algorithm
# Input: function f
#        search range {l0, u0} and accuracy
# Output: minium point and value
#..........................................................

function FibonacciNumber(n::Int64)
        if n == 0 
                return 0
        elseif n == 1
                return 1
        elseif n > 1
                return FibonacciNumber(n-1) + FibonacciNumber(n-2)
        else # n<0
                return FibonacciNumber(n+2) - FibonacciNumber(n+1)
        end
end

# Different convention from FibonacciNumber
# WARNING: n should be positive integers
function FibonacciSequence(n::Int64)
        a = ones(Int64, n)
        for i = 3:n
                a[i] = a[i-1] + a[i-2]
        end
        return a
end


function FibonacciSection(f, l0, u0, accuracy)
    if l0 >= u0
        #println("WARNING: Two bounds have been interchanged!")
        u = l0
        l = u0
    else
        u = u0
        l = l0
    end
    ratio = (u-l)/abs(accuracy)
    maxIter = convert(Int64, floor(-log((sqrt(5) - 1)/2, ratio))) 
    # Construct Fibonacci sequence
    a = ones(Int64, maxIter)
    for i = 3:maxIter
            a[i] = a[i-1] + a[i-2]
            if a[i] > ratio
                    maxIter = i
                    break
            end
    end
    rho = 1 - a[maxIter-1]/a[maxIter]
    # Initial iteration, setup for later iterations
    delta = rho*(u-l)
    u1 = u - delta
    l1 = l + delta
    fu1 = f(u1)
    fl1 = f(l1)
    for i = maxIter:-1:3
        if i > 3
            rho = 1 - a[i-1]/a[i]
        else
            rho = (1 + 2*eps())/2
        end
        if fl1 < fu1 # New range, {l, u1}
           u = u1
           u1 = l1
           fu1 = fl1
           l1 = l + rho*(u1-l)
           fl1 = f(l1)
        else # New range, {l1, u}
           l = l1
           l1 = u1
           fl1 = fu1
           u1 = u - rho*(u-l1)
           fu1 = f(u1)
        end
    end
    # Final result
    return (l+u)/2, f((l+u)/2)
end
    
       
    
#..........................................................
# Test Cases
#..........................................................

FibonacciNumber( 8) ==  21
FibonacciNumber(20) == 6765
FibonacciNumber(-7) == 13
FibonacciNumber(-8) == -21

FibonacciSequence(8)

FibonacciSection(x->x^2, -10, 10, eps())

FibonacciSection(x->x^2/2 - sin(x), -20, 20, eps())

#..........................................................
# Binary Section Algorithm
# Input: function f and its 1st derivative f1
#        search range {l0, u0} and accuracy
# Output: minimmum point and value
#..........................................................

function BinarySection(f, f1, l0, u0, accuracy)
    if l0 >= u0
        #println("WARNING: Two bounds have been interchanged!")
        u = l0
        l = u0
    else
        u = u0
        l = l0
    end
    maxIter = ceil(log2((u-l)/abs(accuracy)))
    m = (l + u)/2
    for i = 1: maxIter
        f1m = f1(m)
        if f1m == 0 # WARNING: OK??
            return m, f(m)
        elseif f1m >0
            u = m
        else
            l = m
        end
        m = (l + u)/2
    end
    return m, f(m)
end

#...................................
# Simple test cases
#...................................

BinarySection(x->x^2, x->2x, -10, 10, eps())

BinarySection(x->x^2/2 - sin(x),
              x->x - cos(x),
              -20, 20, eps())


#..........................................................
# Newton Algorithm
# Input: function f and its 1st and 2nd derivatives f1 and f2
#        start point m0, accuracy and optional max. iterations
# Output: minimmum point and value
#..........................................................

function NewtonMethod(f, f1, f2, m0, accuracy, maxIter::Int64=128)
    m = m0
    for i = 1:maxIter
        fm1 = f1(m)
        fm2 = f2(m)
        if fm2 == 0
            println("ERROR: the 2nd derivative is 0.")
            return m, f(m)
        else
            delta = - fm1/fm2
        end
        m = m + delta
        #println(i, "\t", delta)
        if abs(delta) <= abs(accuracy)
            return m, f(m)
        end
    end
    println("WARNING: accuracy is not reached yet. Please increase iteration numbers")
    return m, f(m)
end

           
#...................................
# Simple test cases
#...................................

NewtonMethod(x->x^2, x->2x, x->2, 10, eps())
NewtonMethod(x->x^2/2 - sin(x),
       x->x - cos(x),
       x-> 1 + sin(x),
       20, eps())

#..........................................................
# Secant Method Algorithm
# Input: function f and its 1st derivative f1
#        start points m0, accuracy and optional max. iterations
# Output: minimmum point and value
#..........................................................

function SecantMethod(f, f1, m0, accuracy, maxIter::Int64=128)
    l = m0
    #Arbitrary
    u = m0*1.1
    #Initiation setup
    f1l = f1(l)
    f1u = f1(u)
    for i = 1:maxIter
        delta = f1u - f1l
        if delta == 0
            println("ERROR: the appr. 2nd derivative is 0.")
            return l, f(l)
        else
            delta = f1l*(l-u)/delta
        end
        u = l
        f1u = f1l
        l = l + delta
        f1l = f1(l)
        #println(i, "\t", delta)
        if abs(delta) <= abs(accuracy)
            return l, f(l)
        end
    end
    println("WARNING: accuracy is not reached yet. Please increase iteration numbers")
    return l, f(l)
end

SecantMethod(x->x^2, x->2x, 10.0, eps())

SecantMethod(x->x^2/2 - sin(x),
       x->x - cos(x),
       20, eps())

