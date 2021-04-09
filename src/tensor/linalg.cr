# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

require "./tensor"
require "./extension"
require "./work"

class Tensor(T)
  # Computes the upper triangle of a `Tensor`.  Zeros
  # out values below the `k`th diagonal
  #
  # Arguments
  # ---------
  # *k* : Int
  #   Diagonal
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int32).ones([3, 3])
  # a.triu!
  # a
  #
  # # [[1, 1, 1],
  # #  [0, 1, 1],
  # #  [0, 0, 1]]
  # ```
  def triu!(k : Int = 0)
    self.each_pointer_with_index do |e, i|
      m = i // @shape[1]
      n = i % @shape[1]
      e.value = m > n - k ? T.new(0) : e.value
    end
  end

  # :ditto:
  def triu(k : Int = 0)
    t = self.dup
    t.triu!(k)
    t
  end

  # Computes the lower triangle of a `Tensor`.  Zeros
  # out values above the `k`th diagonal
  #
  # Arguments
  # ---------
  # *k* : Int
  #   Diagonal
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int32).ones([3, 3])
  # a.tril!
  # a
  #
  # # [[1, 0, 0],
  # #  [1, 1, 0],
  # #  [1, 1, 1]]
  # ```
  def tril!(k : Int = 0)
    self.each_pointer_with_index do |e, i|
      m = i // @shape[1]
      n = i % @shape[1]
      e.value = m < n - k ? T.new(0) : e.value
    end
  end

  # :ditto:
  def tril(k : Int = 0)
    t = self.dup
    t.tril!(k)
    t
  end

  # DOT forms the dot product of two vectors.
  # Uses unrolled loops for increments equal to one.
  #
  # Arguments
  # ---------
  # *u* : Tensor
  #   Right hand side of the dot product
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3, 4, 5].to_tensor
  # a.dot(a) # => 55.0
  # ```
  def dot(u : Tensor(T))
    self.is_vector
    u.is_vector

    blas_call(
      dot,
      @size,
      self.to_unsafe,
      @strides[0],
      u.to_unsafe,
      u.strides[0]
    )
  end

  # Cholesky decomposition.
  #
  # Return the Cholesky decomposition, L * L.H, of the square matrix a, where
  # L is lower-triangular and .H is the conjugate transpose operator (which
  # is the ordinary transpose if a is real-valued). a must be Hermitian
  # (symmetric if real-valued) and positive-definite. Only L is actually
  # returned.
  #
  # Arguments
  # ---------
  # *lower*
  #   Triangular of decomposition to return
  #
  # Examples
  # --------
  # ```
  # t = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]].to_tensor.astype(Float32)
  # t.cholesky
  #
  # # [[ 1.414,    0.0,    0.0],
  # #  [-0.707,  1.225,    0.0],
  # #  [   0.0, -0.816,  1.155]]
  # ```
  def cholesky!(*, lower = true)
    self.is_square_matrix
    self.is_fortran

    char = lower ? 'L' : 'U'
    lapack(potrf, char.ord.to_u8, shape[0], to_unsafe_c, shape[0])
    lower ? tril! : triu!
  end

  # :ditto:
  def cholesky(*, lower = true)
    t = self.dup(Num::ColMajor)
    t.cholesky!
    t
  end

  # Compute the qr factorization of a matrix.
  #
  # Factor the matrix a as qr, where q is orthonormal and r is
  # upper-triangular.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[0, 1], [1, 1], [1, 1], [2, 1]].to_tensor.as_type(Float32)
  # q, r = t.qr
  # puts q
  # puts r
  #
  # # [[   0.0,  0.866],
  # #  [-0.408,  0.289],
  # #  [-0.408,  0.289],
  # #  [-0.816, -0.289]]
  # # [[-2.449, -1.633],
  # #  [   0.0,  1.155],
  # #  [   0.0,    0.0],
  # #  [   0.0,    0.0]]
  # ```
  def qr
    self.is_matrix
    m, n = @shape
    k = {m, n}.min
    a = self.dup(Num::ColMajor)
    tau = Tensor(T).new([k])
    jpvt = Tensor(Int32).new([1])
    lapack(geqrf, m, n, a.to_unsafe_c, m, tau.to_unsafe_c)
    r = a.triu
    lapack(orgqr, m, n, k, a.to_unsafe_c, m, tau.to_unsafe_c)
    {a, r}
  end

  # Singular Value Decomposition.
  #
  # When a is a 2D array, it is factorized as u @ np.diag(s) @ vh = (u * s) @ vh,
  # where u and vh are 2D unitary arrays and s is a 1D array of a’s singular
  # values.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[0, 1], [1, 1], [1, 1], [2, 1]].to_tensor.as_type(Float32)
  # a, b, c = t.svd
  # puts a
  # puts b
  # puts c
  #
  # # [[-0.203749, 0.841716 , -0.330613, 0.375094 ],
  # #  [-0.464705, 0.184524 , -0.19985 , -0.842651],
  # #  [-0.464705, 0.184524 , 0.861075 , 0.092463 ],
  # #  [-0.725662, -0.472668, -0.330613, 0.375094 ]]
  # # [3.02045 , 0.936426]
  # # [[-0.788205, -0.615412],
  # #  [-0.615412, 0.788205 ]]
  # ```
  def svd
    self.is_matrix
    a = dup(Num::ColMajor)
    m, n = a.shape
    mn = {m, n}.min
    mx = {m, n}.max
    s = Tensor(T).new([mn])
    u = Tensor(T).new([m, m])
    vt = Tensor(T).new([n, n])
    lapack(gesdd, 'A'.ord.to_u8, m, n, a.to_unsafe_c, m, s.to_unsafe_c, u.to_unsafe_c, m,
      vt.to_unsafe_c, n, worksize: [{5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn}.max, 8*mn])
    {u.transpose, s, vt.transpose}
  end

  # Compute the eigenvalues and right eigenvectors of a square `Tensor`.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[0, 1], [1, 1]].to_tensor.as_type(Float32)
  # w, v = t.eigh
  # puts w
  # puts v
  #
  # # [-0.618034, 1.61803  ]
  # # [[-0.850651, 0.525731 ],
  # #  [0.525731 , 0.850651 ]]
  # ```
  def eigh
    self.is_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    w = Tensor(T).new([n])
    lapack(
      syev,
      'V'.ord.to_u8,
      'L'.ord.to_u8,
      n,
      a.to_unsafe_c,
      n,
      w.to_unsafe_c,
      worksize: 3 * n - 1
    )
    {w, a}
  end

  # Compute the eigenvalues and right eigenvectors of a square array.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[0, 1], [1, 1]].to_tensor.as_type(Float32)
  # w, v = t.eig
  # puts w
  # puts v
  #
  # # [-0.618034, 1.61803  ]
  # # [[-0.850651, 0.525731 ],
  # #  [-0.525731, -0.850651]]
  # ```
  def eig
    self.is_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    wr = Tensor(T).new([n])
    wl = wr.dup
    vl = Tensor(T).new([n, n], Num::RowMajor)
    vr = wr.dup
    lapack(geev, 'V'.ord.to_u8, 'V'.ord.to_u8, n, a.to_unsafe_c, n, wr.to_unsafe_c,
      wl.to_unsafe_c, vl.to_unsafe_c, n, vr.to_unsafe_c, n, worksize: 3 * n)
    {wr, vl}
  end

  # Compute the eigenvalues of a general matrix.
  #
  # Main difference between eigvals and eig: the eigenvectors aren’t
  # returned.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[0, 1], [1, 1]].to_tensor.as_type(Float32)
  # puts t.eigvalsh
  #
  # # [-0.618034, 1.61803  ]
  # ```
  def eigvalsh
    self.is_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    w = Tensor(T).new([n])
    lapack(syev, 'N'.ord.to_u8, 'L'.ord.to_u8, n, a.to_unsafe_c, n, w.to_unsafe_c, worksize: 3 * n - 1)
    w
  end

  # Compute the eigenvalues of a general matrix.
  #
  # Main difference between eigvals and eig: the eigenvectors aren’t
  # returned.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[0, 1], [1, 1]].to_tensor.as_type(Float32)
  # puts t.eigvals
  #
  # # [-0.618034, 1.61803  ]
  # ```
  def eigvals
    self.is_square_matrix
    a = self.dup(Num::ColMajor)
    n = a.shape[0]
    wr = Tensor(T).new([n])
    wl = wr.dup
    vl = Tensor(T).new([n, n])
    vr = wr.dup
    lapack(geev, 'N'.ord.to_u8, 'N'.ord.to_u8, n, a.to_unsafe_c, n, wr.to_unsafe_c,
      wl.to_unsafe_c, vl.to_unsafe_c, n, vr.to_unsafe_c, n, worksize: 3 * n)
    wr
  end

  # Matrix norm
  #
  # This function is able to return one of eight different matrix norms
  #
  # Arguments
  # ---------
  # *order* : String
  #   Type of norm
  #
  # Examples
  # --------
  # ```
  # t = [[0, 1], [1, 1], [1, 1], [2, 1]].to_tensor.as_type(Float32)
  # t.norm # => 3.6055512
  # ```
  def norm(*, order = 'F')
    self.is_matrix
    a = self.dup(Num::ColMajor)
    m, n = a.shape
    worksize = order == 'I' ? m : 0
    lapack_util(lange, worksize, order.ord.to_u8, m, n, tensor(a.to_unsafe_c), m)
  end

  # Compute the determinant of an array.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[1, 2], [3, 4]].to_tensor.astype(Float32)
  # puts t.det # => -2.0
  # ```
  def det
    self.is_square_matrix
    a = dup(Num::ColMajor)
    m, n = a.shape
    ipiv = Pointer(Int32).malloc(n)

    lapack(getrf, m, n, a.to_unsafe_c, n, ipiv)
    ldet = Num.prod(a.diagonal)
    detp = 1
    n.times do |j|
      if j + 1 != ipiv[j]
        detp = -detp
      end
    end
    ldet * detp
  end

  # Compute the (multiplicative) inverse of a matrix.
  #
  # Given a square matrix a, return the matrix ainv satisfying
  # dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[1, 2], [3, 4]].to_tensor.as_type(Float32)
  # puts t.inv
  #
  # # [[-2  , 1   ],
  # #  [1.5 , -0.5]]
  # ```
  def inv
    self.is_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    ipiv = Pointer(Int32).malloc(n)
    lapack(getrf, n, n, a.to_unsafe_c, n, ipiv)
    lapack(getri, n, a.to_unsafe_c, n, ipiv, worksize: n * n)
    a
  end

  # Solve a linear matrix equation, or system of linear scalar equations.
  #
  # Computes the “exact” solution, x, of the well-determined, i.e., full rank,
  # linear matrix equation ax = b.
  #
  # Arguments
  # ---------
  # *x* : Tensor
  #   Argument with which to solve
  #
  # Examples
  # --------
  # ```
  # a = [[3, 1], [1, 2]].to_tensor.astype(Float32)
  # b = [9, 8].to_tensor.astype(Float32)
  # puts a.solve(b)
  #
  # # [2, 3]
  # ```
  def solve(x : Tensor(T))
    self.is_square_matrix
    a = dup(Num::ColMajor)
    x = x.dup(Num::ColMajor)
    n = a.shape[0]
    m = x.rank > 1 ? x.shape[1] : x.shape[0]
    ipiv = Pointer(Int32).malloc(n)
    lapack(gesv, n, m, a.to_unsafe_c, n, ipiv, x.to_unsafe_c, m)
    x
  end

  # Compute Hessenberg form of a matrix.
  #
  # The Hessenberg decomposition is:
  #
  # ```
  # A = Q H Q^H
  # ```
  #
  # where Q is unitary/orthogonal and H has only zero elements below the first sub-diagonal.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = [[2, 5, 8, 7],
  #      [5, 2, 2, 8],
  #      [7, 5, 6, 6],
  #      [5, 4, 4, 8]].to_tensor.as_type(Float64)
  #
  # puts a.hessenberg
  #
  # # [[2       , -11.6584, 1.42005 , 0.253491],
  # #  [-9.94987, 14.5354 , -5.31022, 2.43082 ],
  # #  [0       , -1.83299, 0.3897  , -0.51527],
  # #  [0       , 0       , -3.8319 , 1.07495 ]]
  # ```
  def hessenberg
    self.is_square_matrix
    a = dup(Num::ColMajor)

    if a.shape[0] < 2
      return a
    end

    n = a.shape[0]
    s = of_real_type(n)
    ilo = 0
    ihi = 0
    lapack(gebal, 'B'.ord.to_u8, n, a.to_unsafe_c, n, ilo, ihi, s.to_unsafe_c)
    tau = Tensor(T).new([n])
    lapack(gehrd, n, ilo, ihi, a.to_unsafe_c, n, tau.to_unsafe_c)
    a.triu(-1)
  end

  # Computes a matrix multiplication between two `Tensors`.  The `Tensor`s
  # must be two dimensional with compatible shapes.  Currently
  # only Float and Complex `Tensor`s are supported, as BLAS is used
  # for this operation
  #
  # Arguments
  # ---------
  # *other* : Tensor(T)
  #   The right hand side of the operation
  #
  # Examples
  # --------
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor.random(0.0...10.0, [3, 3])
  # a.matmul(a)
  #
  # # [[28.2001, 87.4285, 30.5423],
  # #  [12.4381, 30.9552, 26.2495],
  # #  [34.0873, 73.5366, 40.5504]]
  # ```
  def matmul(other : Tensor(T), output : Tensor(T)? = nil)
    self.is_matrix
    other.is_matrix

    unless self.shape[1] == other.shape[0]
      raise Num::Internal::ShapeError.new("Invalid shapes for matrix multiplication: #{@shape}, #{other.shape}")
    end

    if output.nil?
    else
      unless output.shape == [self.shape[0], other.shape[1]]
        raise Num::Internal::ShapeError.new("Invalid output size")
      end
    end

    a = @flags.contiguous? || @flags.fortran? ? self : self.dup(Num::RowMajor)
    b = other.flags.contiguous? || other.flags.fortran? ? other : other.dup(Num::RowMajor)
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    lda = a.flags.contiguous? ? a.shape[1] : a.shape[0]
    ldb = b.flags.contiguous? ? b.shape[1] : b.shape[0]

    if output.nil?
      dest = Tensor(T).new([m, n])
    else
      dest = output
    end

    a_trans = flags.contiguous? ? LibCblas::CblasTranspose::CblasNoTrans : LibCblas::CblasTranspose::CblasTrans
    b_trans = other.flags.contiguous? ? LibCblas::CblasTranspose::CblasNoTrans : LibCblas::CblasTranspose::CblasTrans
    alpha = T.new(1.0)
    c_alpha = T.new(0.0)
    blas(
      ge,
      mm,
      a_trans,
      b_trans,
      m,
      n,
      k,
      blas_const(alpha),
      a.to_unsafe_c,
      lda,
      b.to_unsafe_c,
      ldb,
      blas_const(c_alpha),
      dest.to_unsafe_c,
      dest.shape[1]
    )
    dest
  end

  # Compute tensor dot product along specified axes.
  #
  # Given two tensors, a and b, and an array_like object containing two
  # array_like objects, (a_axes, b_axes), sum the products of a’s and b’s
  # elements (components) over the axes specified by a_axes and b_axes.
  # The third argument can be a single non-negative integer_like scalar,
  # N; if it is such, then the last N dimensions of a and the first N
  # dimensions of b are summed over.
  #
  # Arguments
  # ---------
  # *b* : Tensor
  #   Right hand side of dot products
  # *axes* : Array(Array(Int)) | Array(Int) | Int
  #   Axes of summation
  #
  # Examples
  # --------
  # ```
  # a = Tensor.range(60.0).reshape(3, 4, 5)
  # b = Tensor.range(24.0).reshape(4, 3, 2)
  # puts a.tensordot(b, axes: [[1, 0], [0, 1]])
  #
  # # [[4400, 4730],
  # #  [4532, 4874],
  # #  [4664, 5018],
  # #  [4796, 5162],
  # #  [4928, 5306]]
  # ```
  def tensordot(b : Tensor(T), axes : Array(Array(Int)))
    axes_a, axes_b = axes
    na = axes_a.size
    nb = axes_b.size
    as_ = self.shape
    nda = self.rank
    bs = b.shape
    ndb = b.rank
    equal = na == nb
    na.times do |k|
      if as_[axes_a[k]] != bs[axes_b[k]]
        equal = false
        break
      end
      if axes_a[k] < 0
        axes_a[k] += nda
      end
      if axes_b[k] < 0
        axes_b[k] += ndb
      end
    end
    unless equal
      raise Num::Internal::ShapeError.new("Shape mismatch for sum")
    end
    notin = (0...nda).select do |k|
      !axes_a.includes?(k)
    end
    newaxes_a = notin + axes_a
    n2 = 1
    axes_a.each do |axis|
      n2 *= as_[axis]
    end
    newshape_a = [(notin.map { |ax| as_[ax] }).product, n2]
    olda = notin.map { |ax| as_[ax] }

    notin = (0...ndb).select do |k|
      !axes_b.includes?(k)
    end
    newaxes_b = axes_b + notin
    n2 = 1
    axes_b.each do |axis|
      n2 *= bs[axis]
    end
    newshape_b = [n2, (notin.map { |ax| bs[ax] }).product]
    oldb = notin.map { |ax| bs[ax] }

    at = self.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = at.matmul(bt)
    res.reshape(olda + oldb)
  end

  # :ditto:
  def tensordot(b : Tensor(T), axes : Int)
    axes_a = (-axes...0).to_a
    axes_b = (0...axes).to_a
    self.tensordot(b, [axes_a, axes_b])
  end

  # :ditto:
  def tensordot(b : Tensor(T), axes : Array(Int))
    axes_a, axes_b = axes
    self.tensordot(b, [[axes_a], [axes_b]])
  end

  # Compute the matrix exponential using Pade approximation.
  #
  # Arguments
  # ---------
  # *self*
  #   Matrix of which to compute the exponential
  #
  # Examples
  # --------
  # ```
  # a = [[1.0, 2.0], [-1.0, 3.0]].to_tensor * Complex.new(0, 1)
  # puts a.expm
  #
  # # [[0.426459+1.89218j  , -2.13721+-0.978113j],
  # #  [1.06861+0.489056j  , -1.71076+0.914063j ]]
  # ```
  def expm
    self.is_matrix

    a_l1 = self.norm(order: '1')
    n_squarings = 0

    {% if T == Float64 || T == Complex %}
      if a_l1 < 1.495585217958292e-002
        u, v = self.pade_three
      elsif a_l1 < 2.539398330063230e-001
        u, v = self.pade_five
      elsif a_l1 < 9.504178996162932e-001
        u, v = self.pade_seven
      elsif a_l1 < 2.097847961257068e+000
        u, v = self.pade_nine
      else
        maxnorm = 5.371920351148152
        n_squarings = {0, Math.log2(a_l1 / maxnorm).ceil.to_i}.max
        a = self / 2**n_squarings
        u, v = a.pade_thirteen
      end
      num = u + v
      den = u.map(v) do |i, j|
        -i + j
      end

      r = den.solve(num)

      n_squarings.times do
        r = r.matmul(r)
      end
      r
    {% elsif T == Float32 %}
      if a_l1 < 4.258730016922831e-001
        u, v = self.pade_three
      elsif a_l1 < 1.880152677804762e+000
        u, v = self.pade_five
      else
        maxnorm = 3.925724783138660
        n_squarings = {0, Math.log2(a_l1 / maxnorm).ceil.to_i}.max
        a = self / 2**n_squarings
        u, v = a.pade_thirteen
      end
      num = u + v
      den = u.map(v) do |i, j|
        -i + j
      end

      r = den.solve(num)

      n_squarings.times do
        r = r.matmul(r)
      end
      r
    {% else %}
      {% raise Num::Internal::ShapeError.new("Invalid type #{T} for expm") %}
    {% end %}
  end

  # :nodoc:
  protected def pade_three
    b = [120, 60, 12, 1]
    i, j = @shape
    ident = Tensor(T).eye(i, j)
    a2 = self.matmul(self)

    inter = a2.map(ident) do |i, j|
      b[3] * i + b[1] * j
    end

    u = self.matmul(inter)
    v = a2.map(ident) do |i, j|
      b[2] * i + b[0] * j
    end
    {u, v}
  end

  # :nodoc:
  protected def pade_five
    b = [30240, 15120, 3360, 420, 30, 1]
    i, j = @shape
    ident = Tensor(T).eye(i, j)
    a2 = self.matmul(self)
    a4 = a2.matmul(a2)

    inter = a4.map(a2, ident) do |i, j, k|
      b[5] * i + b[3] * j + b[1] * k
    end

    u = self.matmul(inter)
    v = a4.map(a2, ident) do |i, j, k|
      b[4] * i + b[2] * j + b[0] * k
    end
    {u, v}
  end

  # :nodoc:
  protected def pade_seven
    b = [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1]
    i, j = @shape
    ident = Tensor(T).eye(i, j)
    a2 = self.matmul(self)
    a4 = a2.matmul(a2)
    a6 = a4.matmul(a2)

    u_lhs = a6.map(a4, a2) do |i, j, k|
      b[7] * i + b[5] * j + b[3] * k
    end
    u_rhs = ident.map { |i| b[1] * i }

    u = self.matmul(u_lhs + u_rhs)

    v_lhs = a6.map(a4, a2) do |i, j, k|
      b[6] * i + b[4] * j + b[2] * k
    end

    v_rhs = ident.map { |i| b[0] * i }
    {u, v_lhs + v_rhs}
  end

  # :nodoc:
  protected def pade_nine
    b = [17643225600, 8821612800, 2075673600, 302702400, 30270240,
         2162160, 110880, 3960, 90, 1]

    i, j = @shape
    ident = Tensor(T).eye(i, j)
    a2 = self.matmul(self)
    a4 = a2.matmul(a2)
    a6 = a4.matmul(a2)
    a8 = a6.matmul(a2)

    u_lhs = a8.map(a6, a4) do |i, j, k|
      b[9] * i + b[7] * j + b[5] * k
    end

    u_rhs = a2.map(ident) do |i, j|
      b[3] * i + b[1] * j
    end

    u = self.matmul(u_lhs + u_rhs)

    v_lhs = a8.map(a6, a4) do |i, j, k|
      b[8] * i + b[6] * j + b[4] * k
    end

    v_rhs = a2.map(ident) do |i, j|
      b[2] * i + b[0] * j
    end

    {u, v_lhs + v_rhs}
  end

  # :nodoc:
  protected def pade_thirteen
    b = [64764752532480000, 32382376266240000, 7771770303897600,
         1187353796428800, 129060195264000, 10559470521600, 670442572800,
         33522128640, 1323241920, 40840800, 960960, 16380, 182, 1]

    i, j = @shape
    ident = Tensor(T).eye(i, j)

    a2 = self.matmul(self)
    a4 = a2.matmul(a2)
    a6 = a4.matmul(a2)

    u_dot_first = a6.map(a4, a2) do |i, j, k|
      b[13] * i + b[11] * j + b[9] * k
    end

    u_lhs = a6.map(a4, a2) do |i, j, k|
      b[7] * i + b[5] * j + b[3] * k
    end

    u_rhs = ident.map { |i| b[1] * i }

    u = self.matmul(a6.matmul(u_dot_first) + u_lhs + u_rhs)

    v_dot_first = a6.map(a4, a2) do |i, j, k|
      b[12] * i + b[10] * j + b[8] * k
    end

    v_lhs = a6.map(a4, a2) do |i, j, k|
      b[6] * i + b[4] * j + b[2] * k
    end

    v_rhs = ident.map { |i| b[0] * i }

    v = a6.matmul(v_dot_first) + v_lhs + v_rhs
    {u, v}
  end

  # :nodoc:
  def is_matrix
    unless self.rank == 2
      raise Exception.new
    end
  end

  # :nodoc:
  def is_square_matrix
    unless self.rank == 2 && @shape[0] == @shape[1]
      raise Exception.new
    end
  end

  # :nodoc:
  def is_fortran
    unless @flags.fortran?
      raise Exception.new
    end
  end

  # :nodoc:
  def is_vector
    unless self.rank == 1
      raise Exception.new
    end
  end
end
