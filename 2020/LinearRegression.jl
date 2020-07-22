using LinearAlgebra


function linearsolve(A, b)
  P = inv(A'*A)
  x = P*(A'*b) 
  return (x, P)
end

function linearsolve(x, P, a, b)
  Pa = P*a   #Vector
  Pn = P .- Pa*a'*P/(1+a'*Pa)
  xn = x .+ (b - a'*x) .* Pn*a
  return (xn, Pn)
end

linearsolve([ 1  2  3  4; 
	      7  6  5  8; 
	      10 8  9 12;
	     13  15 14 16;
	     19  20 17 18], 
	    [10, 21, 22, 23, 24])

x, P =
linearsolve([ 1  2  3  4; 
	      7  6  5  8; 
	      10 8  9 12;
	     13  15 14 16], 
	    [10, 21, 22, 23])
x, P = linearsolve(x, P, 
		   [19, 20, 17, 18],
		   24)


function LinearSolve(A,
		     b;
		     iterativeQ = true,
		     constantQ = false
		     )
  #Check if the dimensions match
  if !(ndims(A) == 2 && ndims(b) == 1) # MUST be double '&'
     error("A should be a matrix (dims = 2) and b should be a vector (dim = 1)")
  end
  A0 = A
  r, c = size(A0)
  l = length(b)
  if constantQ
     A0 = hcat(A0, ones(r))
  end
  if r != l
     error("# of rows in A should match with # of elements in b")
  end
  if r < c 
     error("Not implemented yet!")
  end
  if !iterativeQ
     return linearsolve(A0, b)
  end
  x, P = linearsolve(A0[1:c, 1:end], b[1:c])
  for i in collect(c+1:r)
     x, P = linearsolve(x, P, A0[i,1:end], b[i])
  end
  return (x, P)
end

# Basic Test Case
x, P =
LinearSolve([ 1  2  3  4; 
	      7  6  5  8; 
	      10 8  9 12;
	     13  15 14 16;
	     19  20 17 18;
	     3    8 20 12], 
	    [10, 21, 22, 23, 24, 13], 
	    iterativeQ = false)

x, P =
LinearSolve([ 1  2  3  4; 
	      7  6  5  8; 
	      10 8  9 12;
	     13  15 14 16;
	     19  20 17 18;
	     3    8 20 12], 
	    [10, 21, 22, 23, 24, 13], 
	    constantQ =true, 
	    iterativeQ = false)


x, P =
LinearSolve([ 1  2  3  4; 
	      7  6  5  8; 
	      10 8  9 12;
	     13  15 14 16;
	     19  20 17 18;
	     3   8  20 12], 
	    [10, 21, 22, 23, 24, 13])

