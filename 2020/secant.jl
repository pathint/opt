using LinearAlgebra

# Secant method to
# optimize step size, ϕ(α) = f(x_k + α*d), 
# in high-dimensional optimization problems
# α = α + (- ϕ'(α)/ϕ''(α))
# ϕ'' = (ϕ'(α_k) - ϕ'(α_(k-1))/(a_k - α_(k-1))
# |ϕ'| < ϵ
# ϕ'(α) = d'*g(α)
#
function secant_alpha(g,
		      d,
		      x0, g0;
		      ϵ = 0.001,
		      verbose = false,
		      maxIter = 128)
   αk = 0
   αn = 0.01
   xk = x0 
   xn = x0 .+ αn .* d
   gk = g0
   gn = g(xn...)
   dk = d'*gk # ϕ'(αk)
   dn = d'*gn # ϕ'(αn)
   for i in 1:maxIter
      #Test convergence
      if norm(dn) < ϵ
         return αn, αn .* d, xn, gn
      end
      if verbose
	 println("i:", i, "\tα:", αn)
      end
      δ = -dn*(αn - αk)/(dn - dk)  
      #Save the current iteration step
      αk = αn
      xk = xn
      gk = gn
      dk = dn
      #Update
      αn = αn + δ
      xn = x0 .+ αn*d
      gn = g(xn...)
      dn = d'*gn
   end
   println("WARN: max iterations ", maxIter, " have been exceeded before convergence.")
   return αn, αn .* d, xn, gn
end

#Test cases

f = (x, y) -> x^2 + 10*y^2
g = (x, y) -> [2x,
	      20y]
x0 = [1, 1]
-g(x0...)

secant_alpha(f, g, -g(x0...), 
	     x0, f(x0...), g(x0...), verbose = true)
