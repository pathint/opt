using LinearAlgebra

function Kaczmarz(A,
		  b,
		  x;
		  μ = 1,
		  ϵ = 0.001)
  #Check if the dimensions match
  if !(ndims(A) == 2 && ndims(b) == 1) # MUST be double '&'
     error("A should be a matrix (dims = 2) and b should be a vector (dim = 1)")
  end
  r, c = size(A)
  l = length(b)
  if l != r
     error("# of rows in A should match with # of elements in b")
  end
  if r > c 
     error("Kaczmarz algorithm is not applicable!")
  end
  xk = x
  xn = x
  i = 0
  A0 = mapslices(r -> r .* (1 /(r'*r)), A, dims=[2])
  pts = []
  push!(pts, xn)
  while i == 0 || norm(xn .- xk) > ϵ
      i  = i + 1
      for j in 1:r
         xk = xn
	 xn = xn .+ μ*(b[j] - A[j, 1:end]'*xn) .* A0[j,1:end]
	 push!(pts, xn)
      end
  end
  return (xn, pts)
end

# Basic Test Case
x, p = 
Kaczmarz([1 -1;
	  0  1],
	 [2, 3],
	 [0, 0])

x, p =
Kaczmarz([1 2],
	 [3],
	 [0, 0])

