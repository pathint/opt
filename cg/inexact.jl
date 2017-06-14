###########################################################
#
# Collection of Optimization/Search Algorithms
# Inexact Line Search Algorithms for Multivariate Function
# to solve a in f(x... + a*d...) to minimize f(x... + a*d...)
# Here ... represents vectors.
# This optimization problems appears in steepest descent,
# conjugate gradient, quasi-Newton and other algorithms
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
# *** FOR TEACHING PURPOSE ONLY ***
###########################################################

#..........................................................
# Goldstein's Rules
# f(x... + a*d...) <= f(x...) +     rho*a*g'*d
# f(x... + a*d...) >= f(x...) + (1-rho)*a*g'*d 
# 0 < rho <1/2
# Input: function f,
#        point x and search direction d
#        the value of f and f' at x
#        optional: search range
# Output: minimum point and value
#..........................................................

function Goldstein(f,   # f(x), scalar function
                   x,   # current x, vector
                   d,   # Search direction, vector
                   fa0, # f(x...), scalar, where a =0, redundant
                   ga0, # D[f, {x, 1}], vector, where a = 0
                   u0::Float64 = 1.0, # max a
                   l0::Float64 = 0.0; # min a
                   rho::Float64 = 0.4,
                   iterations::Int64=128) 
    u = u0
    l = l0
    a = 0
    gd = ga0'*d # 1st derivative of f w.r.t `a` in the `d` direction
    fa = fa0
    for i = 1:iterations #fa >= fa0 + rho*a*gd || fa <= fa0 + (1-rho)*a*gd
        a = (l+u)/2
        fa= f((x .+ a*d)...)
        if l==u
            return a, fa
        end
        if fa <= fa0 + rho*a*gd
            if fa >= fa0 + (1-rho)*a*gd
                return a, fa
            else
                l = a
            end
        else
            u = a
        end
    end
    println("WARNING: Goldstein line search terminal conditions may not be met.")
    return a, fa
end


#...................................
# Simple test cases
#...................................

# f(x1, x2) = x1^2 + (1/2)x2^2+3
# x= [1, 2], d = -g = [-2, 2]
# Hessian = [2 0; 0 1]
# a = -g'*d/(d'*H*d) = 2/3
             
map(u->[u, 
 Goldstein((x1, x2)-> x1^2+0.5x2^2+3,
           [1, 2],
           [-2, -2],
           6,
           [2, 2], u, rho=0.2)...], 1.0:0.1:3.0)

map(u->[u, 
 Goldstein((x1, x2)-> x1^2+0.5x2^2+3,
           [1, 2],
           [-2, -2],
           6,
           [2, 2], u, rho=0.4)...], 1.0:0.1:3.0)

map(u->[u, 
 Goldstein((x1, x2)-> x1^2+0.5x2^2+3,
           [1, 2],
           [-2, -2],
           6,
           [2, 2], u, rho=0.1)...], 1.0:0.1:3.0)

#..........................................................
# Wolfe's Rules
# f(x... + a*d...) <= f(x...) + rho*a*g'*d 
# g(x... + a*d...)'*d >= sigma*g'*d 
# 0 < rho <1/2, rho < sigma < 1
# Input: function f, 1st derivative g,
#        point x and search direction d
#        the value of f and f' at x
#        optional: search range
# Output: minimum point and value
#..........................................................

function Wolfe(f,   # f(x), scalar function
               g,   # f'(x), gradient, vector-valued function
               x,   # current x, vector
               d,   # Search direction, vector
               fa0, # f(x...), scalar, where a =0, redundant
               ga0, # D[f, {x, 1}], vector, where a = 0
               u0::Float64 = 1.0, # max a
               l0::Float64 = 0.0; # min a
               rho::Float64 = 0.4,
               sigma::Float64 = 0.6,
               iterations::Int64 = 256) 
    u = u0/vecnorm(d)
    l = l0
    a = u
    fa1= fa0
    gd1= ga0'*d # 1st derivative of f w.r.t `a` in the `d` direction
    fa = f((x .+ a*d)...)
    for i = 1:iterations  #fa >= fa0 + rho*a*gd || fa <= fa0 + (1-rho)*a*gd
        fa = f((x .+ a*d)...)
        #if l==u
        #    return a, fa
        #end
        if fa <= fa1 + rho*a*gd1
            ga = g((x .+ a*d)...)
            gd = ga'*d
            if gd >= sigma*gd1
                return a, fa
            else
                o = a + (a-l)*gd/(gd1-gd)
                l = a
                fa1 = fa
                a = o
                gd1 = gd
            end
        else
            #2nd-order interpolation
            u = a
            a = l + 0.5*(a-l)/(1+(fa1 -fa)/((a-l)*gd1))
        end
    end
    println("WARNING: Wolfe line search terminal conditions may not be met.")
    return a, fa
end


#...................................
# Simple test cases
#...................................

# f(x1, x2) = x1^2 + (1/2)x2^2+3
# x= [1, 2], d = -g = [-2, 2]
# Hessian = [2 0; 0 1]
# a = -g'*d/(d'*H*d) = 2/3
             
map(u->[u, 
 Wolfe((x1, x2)-> x1^2+0.5x2^2+3,
       (x1, x2)->[2x1
                  x2],
           [1, 2],
           [-2, -2],
           6,
           [2, 2], u)...], 0.8:0.1:12.0)

#..........................................................
# Armijo's Rules
# f(x... + a*d...) <= f(x...) +     rho*a*g'*d
# //f(x... + a*d...) >= f(x...) + (1-rho)*a*g'*d 
# 0 < rho <1/2
# Input: function f,
#        point x and search direction d
#        the value of f and f' at x
#        optional: search range
# Output: minimum point and value
#..........................................................

function Armijo(f,   # f(x), scalar function
                x,   # current x, vector
                d,   # Search direction, vector
                fa0, # f(x...), scalar, where a =0, redundant
                ga0, # D[f, {x, 1}], vector, where a = 0
                u0::Float64 = 1.0, # max a
                l0::Float64 = 0.0; # min a
                rho::Float64 = 0.4,
                tau::Float64 = 0.5,
                iterations::Int64 = 128) 
    u = u0/vecnorm(d)
    l = l0
    a = u
    gd = ga0'*d # 1st derivative of f w.r.t `a` in the `d` direction
    fa = f((x .+ a*d)...)
    i = 0
    while  fa >= fa0 + rho*a*gd && a >= l #|| fa <= fa0 + (1-rho)*a*gd
        i += 1
        #a = tau*a # shrinking by `tau` times
        #2nd order interpolation
        a = 0.5*gd*a^2/(fa0+gd*a-fa)
        if a > u0
            a = tau*u0
        end
        fa = f((x .+ a*d)...)
        if i > iterations
            println("WARNING: Armijo line search terminal conditions may not be met.")
            break
        end
    end
    return a, fa
end

#...................................
# Simple test cases
#...................................

# f(x1, x2) = x1^2 + (1/2)x2^2+3
# x= [1, 2], d = -g = [-2, 2]
# Hessian = [2 0; 0 1]
# a = -g'*d/(d'*H*d) = 2/3
             
map(u->[u, 
 Armijo((x1, x2)-> x1^2+0.5x2^2+3,
           [1, 2],
           [-2, -2],
           6,
           [2, 2], u, rho=0.2)...], 1.0:0.1:3.0)

map(u->[u, 
 Armijo((x1, x2)-> x1^2+0.5x2^2+3,
           [1, 2],
           [-2, -2],
           6,
           [2, 2], u, rho=0.4)...], 1.0:0.1:3.0)

map(u->[u, 
 Armijo((x1, x2)-> x1^2+0.5x2^2+3,
           [1, 2],
           [-2, -2],
           6,
           [2, 2], u, rho=0.45)...], 1.0:0.1:3.0)

