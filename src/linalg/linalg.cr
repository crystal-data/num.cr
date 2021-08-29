# Copyright (c) 2021 Crystal Data Contributors
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

class Tensor(T, S)
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
    assert_square_matrix
    assert_fortran

    char = lower ? 'L' : 'U'
    lapack(potrf, char.ord.to_u8, shape[0], to_unsafe, shape[0])
    lower ? tril! : triu!
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
    self.assert_is_matrix
    m, n = @shape
    k = {m, n}.min
    a = self.dup(Num::ColMajor)
    tau = Tensor(T, S).new([k])
    jpvt = Tensor(Int32, CPU(Int32)).new([1])
    lapack(geqrf, m, n, a.to_unsafe, m, tau.to_unsafe)
    r = a.triu
    lapack(orgqr, m, n, k, a.to_unsafe, m, tau.to_unsafe)
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
    self.assert_is_matrix
    a = self.dup(Num::ColMajor)
    m, n = a.shape
    mn = {m, n}.min
    mx = {m, n}.max
    s = Tensor(T, S).new([mn])
    u = Tensor(T, S).new([m, m])
    vt = Tensor(T, S).new([n, n])
    lapack(gesdd, 'A'.ord.to_u8, m, n, a.to_unsafe, m, s.to_unsafe, u.to_unsafe, m,
      vt.to_unsafe, n, worksize: [{5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn}.max, 8*mn])
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
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    w = Tensor(T, S).new([n])
    lapack(
      syev,
      'V'.ord.to_u8,
      'L'.ord.to_u8,
      n,
      a.to_unsafe,
      n,
      w.to_unsafe,
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
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    wr = Tensor(T, S).new([n])
    wl = wr.dup
    vl = Tensor(T, S).new([n, n])
    vr = wr.dup
    lapack(geev, 'V'.ord.to_u8, 'V'.ord.to_u8, n, a.to_unsafe, n, wr.to_unsafe,
      wl.to_unsafe, vl.to_unsafe, n, vr.to_unsafe, n, worksize: 3 * n)
    {wr, vl}
  end

  # Compute the eigenvalues of a symmetric matrix.
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
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    w = Tensor(T, S).new([n])
    lapack(syev, 'N'.ord.to_u8, 'L'.ord.to_u8, n, a.to_unsafe, n, w.to_unsafe, worksize: 3 * n - 1)
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
    self.assert_square_matrix
    a = self.dup(Num::ColMajor)
    n = a.shape[0]
    wr = Tensor(T, S).new([n])
    wl = wr.dup
    vl = Tensor(T, S).new([n, n])
    vr = wr.dup
    lapack(geev, 'N'.ord.to_u8, 'N'.ord.to_u8, n, a.to_unsafe, n, wr.to_unsafe,
      wl.to_unsafe, vl.to_unsafe, n, vr.to_unsafe, n, worksize: 3 * n)
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
    self.assert_is_matrix
    a = self.dup(Num::ColMajor)
    m, n = a.shape
    result = Tensor(T, S).new([1])
    worksize = order == 'I' ? m : 0
    r = lapack_util(lange, worksize, order.ord.to_u8, m, n, tensor(a.to_unsafe), m)
    result.to_unsafe.value = r
    result
  end

  # Compute the determinant of an array.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = [[1, 2], [3, 4]].to_tensor.as_type(Float32)
  # puts t.det # => -2.0
  # ```
  def det
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    m, n = a.shape
    ipiv = Pointer(Int32).malloc(n)

    result = Tensor(T, S).new([1])

    lapack(getrf, m, n, a.to_unsafe, n, ipiv)

    ldet = T.new(1)
    a.diagonal.each do |el|
      ldet *= el
    end

    detp = 1
    n.times do |j|
      if j + 1 != ipiv[j]
        detp = -detp
      end
    end

    result.to_unsafe.value = ldet * detp
    result
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
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    ipiv = Pointer(Int32).malloc(n)
    lapack(getrf, n, n, a.to_unsafe, n, ipiv)
    lapack(getri, n, a.to_unsafe, n, ipiv, worksize: n * n)
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
  # a = [[3, 1], [1, 2]].to_tensor.as_type(Float32)
  # b = [9, 8].to_tensor.as_type(Float32)
  # puts a.solve(b)
  #
  # # [2, 3]
  # ```
  def solve(x : Tensor(T, S))
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    x = x.dup(Num::ColMajor)
    n = a.shape[0]
    m = x.rank > 1 ? x.shape[1] : x.shape[0]
    ipiv = Pointer(Int32).malloc(n)
    lapack(gesv, n, m, a.to_unsafe, n, ipiv, x.to_unsafe, m)
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
  def dot(u : Tensor(T, S))
    self.assert_is_vector
    u.assert_is_vector
    result = Tensor(T, S).new([1])

    {% if S < OCL %}
      blast(dot, @size, result.to_unsafe, 0, self.to_unsafe, @offset, @strides[0], u.to_unsafe, u.offset, u.strides[0])
    {% else %}
      dotvalue = blas_call(
        dot,
        @size,
        self.to_unsafe,
        @strides[0],
        u.to_unsafe,
        u.strides[0]
      )
      result.to_unsafe.value = dotvalue
    {% end %}
    result
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
  def matmul(other : Tensor(T, S), output : Tensor(T, S)? = nil)
    self.assert_is_matrix
    other.assert_is_matrix

    # unless self.shape[1] == other.shape[0]
    #   raise "Invalid shapes for matrix multiplication: #{@shape}, #{other.shape}"
    # end

    if output.nil?
    else
      unless output.shape == [self.shape[0], other.shape[1]]
        raise "Invalid output size"
      end
    end

    {% if S < OCL %}
      unless (self.is_c_contiguous && other.is_c_contiguous) || (self.is_f_contiguous || other.is_f_contiguous)
        raise "Inputs must be contiguous"
      end
      a = self
      b = other

      m = a.shape[0]
      n = b.shape[1]
      k = a.shape[1]
      lda = self.is_c_contiguous ? a.shape[1] : a.shape[0]
      ldb = other.is_c_contiguous ? b.shape[1] : b.shape[0]

      if output.nil?
        result = Tensor(T, S).new([m, n])
      else
        result = output
      end

      blast(
        gemm,
        self.is_c_contiguous ? LibBlast::CLBlastLayout::CLBlastLayoutRowMajor : LibBlast::CLBlastLayout::CLBlastLayoutColMajor,
        LibBlast::CLBlastTranspose::CLBlastTransposeNo,
        LibBlast::CLBlastTranspose::CLBlastTransposeNo,
        m,
        n,
        k,
        T.new(1.0),
        self.to_unsafe,
        @offset,
        k,
        other.to_unsafe,
        other.offset,
        n,
        0.0,
        result.to_unsafe,
        0,
        n
      )
      result
    {% else %}
      a = self.is_c_contiguous || self.is_f_contiguous ? self : self.dup(Num::RowMajor)
      b = other.is_c_contiguous || other.is_f_contiguous ? other : other.dup(Num::RowMajor)
      m = a.shape[0]
      n = b.shape[1]
      k = a.shape[1]
      lda = a.is_c_contiguous ? a.shape[1] : a.shape[0]
      ldb = b.is_c_contiguous ? b.shape[1] : b.shape[0]

      if output.nil?
        dest = Tensor(T, S).new([m, n])
      else
        dest = output
      end

      a_trans = a.is_c_contiguous ? LibCblas::CblasTranspose::CblasNoTrans : LibCblas::CblasTranspose::CblasTrans
      b_trans = b.is_c_contiguous ? LibCblas::CblasTranspose::CblasNoTrans : LibCblas::CblasTranspose::CblasTrans
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
        a.to_unsafe,
        lda,
        b.to_unsafe,
        ldb,
        blas_const(c_alpha),
        dest.to_unsafe,
        dest.shape[1]
      )
      dest
    {% end %}
  end

  private def assert_square_matrix
    raise "Matrix must be square" unless rank == 2 && @shape[0] == @shape[1]
  end

  private def assert_fortran
    raise "Matrix must be fortran contiguous" unless self.is_f_contiguous
  end

  def assert_is_vector
    raise "Inputs must be vectors" unless self.rank == 1
  end

  def assert_is_matrix
    raise "Input must be a matrix" unless self.rank == 2
  end
end
