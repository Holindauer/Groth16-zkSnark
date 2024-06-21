# Groth16-zkSnark
Groth16 zkSnark Implementation for Valid Witness to Polynomial Constraint


# Qudratic Arithmetic Programs (QAP)

### R1CS

Computation can be represented using a Rank 1 Constraint System (R1CS) as $Ls \circ Rs = Os$ where $L, R$ are matrices representing the left and right hand sides of each quadratic constraint, $O$ is a matrix representing the output of each quadratic constraint, and $s$ is the solution containing witness vector. 

In order to make this representation succinct, we can transform the R1CS into a QAP.

For example, consider if $Ls$ is:

```math
Ls = \begin{bmatrix}
0 0 1 0 0 0 \\ 0 0 0 0 1 0 \\ 0 0 1 0 0 0 
\end{bmatrix}  

\begin{bmatrix} 1 \\ 18 \\ 2 \\ 3 \\ 4 \\ 12 \end{bmatrix} 

= \begin{bmatrix} 2 \\ 4 \\ 2 \end{bmatrix}
```

### Reframing our R1CS Representation

$Ls$ can be rewritten as a sum of hadamard products of column vectors as follows:

```math

Ls = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} 
\circ \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} + 

\begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
\circ \begin{bmatrix} 18 \\ 18 \\ 18 \end{bmatrix} +

\begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}
\circ \begin{bmatrix} 2 \\ 2 \\ 2 \end{bmatrix} +

\begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
\circ \begin{bmatrix} 3 \\ 3 \\ 3 \end{bmatrix} +

\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
\circ \begin{bmatrix} 4 \\ 4 \\ 4 \end{bmatrix} +

\begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
\circ \begin{bmatrix} 12 \\ 12 \\ 12 \end{bmatrix}

= \begin{bmatrix} 2 \\ 4 \\ 2 \end{bmatrix}

```

### Ring Homomorphism and Lagrange Interpolation

 There exists a Ring Homomorphism from column vectors of dimension n with real number elements to polynomials with real coefficients. 
 
 Given $n$ points on a cartesian $(x, y)$ plane, they can be uniquely interpolated by a polynomial of degree $n - 1$. If the degree is not constrained, an infinite number of polynomials of degree $n - 1$ or higher can interpolate those points.

 For example: 

 Given the following column vector:
 
 ```math
 v = \begin{bmatrix} 4 \\ 12 \\ 6 \end{bmatrix}
 ```
 
 Using lagrange polynomial interpolation, we can encode the vector as:
 
 $$p(x) = -7x^2 + 29x - 18$$

 $$p(1) = 4$$
 $$p(2) = 12$$
 $$p(3) = 6$$


The ring homomorphism is from column vector addition and multiplication to polynomial addition and multiplication. 

### R1CS to QAP Transformation Function


To transform our R1CS into a QAP, we will define a transformation function for each of the matrices $L, R, O$.


If we consider the below representation of our R1CS, we can define the transformation function $\varphi$ as the lagrange interpolation of each column of $L, R, O$.


$$ U = \varphi(L) = [u_1(x), u_2(x),..., u_n(x)] L \in \mathbb{F_p}^{n \times m} $$
$$ V = \varphi(R) = [v_1(x), v_2(x),..., v_n(x)] V \in \mathbb{F_p}^{n \times m} $$
$$ W = \varphi(O) = [w_1(x), w_2(x),..., w_n(x)] W \in \mathbb{F_p}^{n \times m} $$
$$ s = a = (a_1, ... , a_n), a_1 = 1, a{}_{i, i>1} \in \mathbb{F_p} $$

Note that the interpolation of each row of the column vector maps to the dimension of that row - 1. And all arithmatic operations are done in the field mod a prime $\mathbb{F_p}$.

The following is equivalent to the R1CS representation:

$$ (U \cdot s)(V \cdot s) = W \cdot s $$

$$ \sum_{i=0}^m a_iu_i(x) \sum_{i=0}^m a_iv_i(x) = \sum_{i=0}^m a_iw_i(x) $$

However, the dot products above will result in a single polynomial when computed, as opposed to very large matrices.

### Balancing the QAP

There is a catch to the above statement. Because we are multiplying $U \cdot s$ and $V \cdot s$, the degree of the resulting polynomial will not match that of $W \cdot s$. The interpolation wrt to the R1CS will still hold, but in order maintain the equality of the polynomial equation. 

For example, consider the following setup for $U, V$

$$
(U \cdot s) = x^2 + x + 1
$$
$$
(V \cdot s) = 3x^2 -2x +1
$$
$$
(U \cdot s)(V \cdot s) = 3x^4 + x^3 + 2x^2 - x + 1
$$

In this situation, $W \cdot s$ will only a degree of 2, breaking the equality on the polynomial side of things. We need to introduce another term into the right hand side of the equation.

First, lets consider our R1CS again. We are not trying to change this representation. If we add a term that is the zero vector, the equation will still hold:

$$ (U \cdot s)(V \cdot s) = W \cdot s + 0 $$

The $0$ term is known as a balancing term. Note that while on the R1CS side, this is a zero vector, however because we can interpolate the polynomial in an infinite number of ways, we can make the polynomial match the degree of the lhs.

In our example, because we are interpolating the polynomials over x = {1, 2, 3}, we can construct a polynomial $h(x)t(x)$ where:
$$
t(x) = (x - 1)(x - 2)(x - 3)
$$

$h(x)$ can then be computed via the following algebraic manipulation:

$$
(U \cdot s)(V \cdot s) - (W \cdot s) = 0
$$
$$
(U \cdot s)(V \cdot s) - (W \cdot s) = h(x)t(x)
$$
$$
\frac{(U \cdot s)(V \cdot s) - (W \cdot s)}{t(x)} = h(x)
$$

It should be noted that $t(x)$ is a public polynomial. This means that in a ZKP, the prover cannot just make up some $h(x)$ term that will make the equation hold. The prover must derive $h(x)$ else the verifier will not be able to verify the proof.
# Sources

https://www.rareskills.io/post/quadratic-arithmetic-program
