require "./extension"
require "./tensor"
require "./creation"

class Bottle::Tensor(T) < Bottle::BaseArray(T)
  private def raise_fortran_inplace(flags)
    unless flags.fortran?
      raise Exceptions::LinAlgError.new("Tensor must be Fortran Contiguous to apply the operation in-place")
    end
  end

  private def assert_square_matrix(a)
    if a.ndims != 2 || a.shape[0] != a.shape[1]
      raise Exceptions::ShapeError.new("Input must be a square matrix")
    end
  end

  private def assert_matrix(a)
    if a.ndims != 2
      raise Exceptions::ShapeError.new("Input must be a matrix")
    end
  end

  private def vector_shape_match(a, b)
    if a.shape != b.shape
      raise Exceptions::ShapeError.new("Shapes do not match")
    end
  end

  private def matrix_shape_match(a, b)
    if a.shape[-1] != b.shape[-2]
      raise Exceptions::ShapeError.new("Matrices are not compatible")
    end
  end

  private def insist_1d(a, b)
    if a.ndims != 1 || b.ndims != 1
      raise Exceptions::ShapeError.new("Inputs must be 1D")
    end
  end

  # Cholesky decomposition.
  #
  # Return the Cholesky decomposition, L * L.H, of the square matrix a, where
  # L is lower-triangular and .H is the conjugate transpose operator (which
  # is the ordinary transpose if a is real-valued). a must be Hermitian
  # (symmetric if real-valued) and positive-definite. Only L is actually
  # returned.
  #
  # ```crystal
  # t = Tensor.from_array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]).astype(Float32)
  # puts t.cholesky
  #
  # # Tensor([[ 1.414,    0.0,    0.0],
  # #         [-0.707,  1.225,    0.0],
  # #         [   0.0, -0.816,  1.155]])
  # ```
  def cholesky!(*, lower = true)
    assert_square_matrix(self)
    raise_fortran_inplace(flags)
    char = lower ? 'L' : 'U'
    lapack(potrf, char.ord.to_u8, shape[0], to_unsafe, shape[0])
    lower ? tril! : triu!
  end

  # Cholesky decomposition.
  #
  # Return the Cholesky decomposition, L * L.H, of the square matrix a, where
  # L is lower-triangular and .H is the conjugate transpose operator (which
  # is the ordinary transpose if a is real-valued). a must be Hermitian
  # (symmetric if real-valued) and positive-definite. Only L is actually
  # returned.
  #
  # ```crystal
  # t = Tensor.from_array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]).astype(Float32)
  # puts t.cholesky
  #
  # # Tensor([[ 1.414,    0.0,    0.0],
  # #         [-0.707,  1.225,    0.0],
  # #         [   0.0, -0.816,  1.155]])
  # ```
  def cholesky(*, lower = true)
    ret = dup('F')
    ret.cholesky!
    ret
  end

  private def qr_setup(a, m, n, k)
    tau = Tensor(T).new([k])
    jpvt = Tensor(Int32).new([1])
    lapack(geqrf, m, n, a.to_unsafe, m, tau.to_unsafe)
    {tau, jpvt}
  end

  # Compute the qr factorization of a matrix.
  #
  # Factor the matrix a as qr, where q is orthonormal and r is
  # upper-triangular.
  #
  # ```crystal
  # t = Tensor.from_array([[0, 1], [1, 1], [1, 1], [2, 1]]).astype(Float32)
  # q, r = t.qr
  # puts q
  # puts r
  #
  # # Tensor([[   0.0,  0.866],
  # #         [-0.408,  0.289],
  # #         [-0.408,  0.289],
  # #         [-0.816, -0.289]])
  # # Tensor([[-2.449, -1.633],
  # #         [   0.0,  1.155],
  # #         [   0.0,    0.0],
  # #         [   0.0,    0.0]])
  # ```
  def qr
    assert_matrix(self)
    m, n = shape
    k = {m, n}.min
    a = dup('F')
    tau = qr_setup(a, m, n, k)[0]
    r = Creation.triu(a)
    lapack(orgqr, m, n, k, a.to_unsafe, m, tau.to_unsafe)
    {a, r}
  end

  # Singular Value Decomposition.
  #
  # When a is a 2D array, it is factorized as u @ np.diag(s) @ vh = (u * s) @ vh,
  # where u and vh are 2D unitary arrays and s is a 1D array of a’s singular
  # values.
  #
  # ```crystal
  # t = Tensor.from_array([[0, 1], [1, 1], [1, 1], [2, 1]]).astype(Float32)
  # a, b, c = t.svd
  # puts a
  # puts b
  # puts c
  #
  # # Tensor([[-0.204,  0.842, -0.331,  0.375],
  # #         [-0.465,  0.185,   -0.2, -0.843],
  # #         [-0.465,  0.185,  0.861,  0.092],
  # #         [-0.726, -0.473, -0.331,  0.375]])
  # # Tensor([ 3.02, 0.936])
  # # Tensor([[-0.788, -0.615],
  # #         [-0.615,  0.788]])
  # ```
  def svd
    assert_matrix(self)
    a = dup('F')
    m, n = a.shape
    mn = {m, n}.min
    mx = {m, n}.max
    s = Tensor(T).new([mn])
    u = Tensor(T).new([m, m])
    vt = Tensor(T).new([n, n])
    lapack(gesdd, 'A'.ord.to_u8, m, n, a.to_unsafe, m, s.to_unsafe, u.to_unsafe, m,
      vt.to_unsafe, n, worksize: [{5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn}.max, 8*mn])
    {u.transpose, s, vt.transpose}
  end

  # Compute the eigenvalues and right eigenvectors of a square array.
  #
  # ```crystal
  # t = Tensor.from_array([[0, 1], [1, 1]]).astype(Float32)
  # w, v = t.eigh
  # puts w
  # puts v
  #
  # # Tensor([-0.618,  1.618])
  # # Tensor([[-0.851,  0.526],
  # #         [ 0.526,  0.851]])
  # ```
  def eigh
    assert_square_matrix(self)
    a = dup('F')
    n = a.shape[0]
    w = Tensor(T).new([n])
    lapack(syev, 'V'.ord.to_u8, 'L'.ord.to_u8, n, a.to_unsafe, n, w.to_unsafe, worksize: 3 * n - 1)
    {w, a}
  end

  # Compute the eigenvalues and right eigenvectors of a square array.
  #
  # ```crystal
  # t = Tensor.from_array([[0, 1], [1, 1]]).astype(Float32)
  # w, v = t.eig
  # puts w
  # puts v
  #
  # # Tensor([-0.618,  1.618])
  # # Tensor([[-0.851,  0.526],
  # #         [ 0.526,  0.851]])
  # ```
  def eig
    assert_square_matrix(self)
    a = dup('F')
    n = a.shape[0]
    wr = Tensor(T).new([n])
    wl = wr.dup
    vl = Tensor(T).new([n, n], ArrayFlags::Fortran)
    vr = wr.dup
    lapack(geev, 'V'.ord.to_u8, 'V'.ord.to_u8, n, a.to_unsafe, n, wr.to_unsafe,
      wl.to_unsafe, vl.to_unsafe, n, vr.to_unsafe, n, worksize: 3 * n)
    {wr, vl}
  end

  # Compute the eigenvalues of a general matrix.
  #
  # Main difference between eigvals and eig: the eigenvectors aren’t
  # returned.
  #
  # ```
  # t = Tensor.from_array([[0, 1], [1, 1]]).astype(Float32)
  # puts t.eigvalsh
  #
  # # Tensor([-0.618,  1.618])
  # ```
  def eigvalsh
    assert_square_matrix(self)
    a = dup('F')
    n = a.shape[0]
    w = Tensor(T).new([n])
    lapack(syev, 'N'.ord.to_u8, 'L'.ord.to_u8, n, a.to_unsafe, n, w.to_unsafe, worksize: 3 * n - 1)
    w
  end

  # Compute the eigenvalues of a general matrix.
  #
  # Main difference between eigvals and eig: the eigenvectors aren’t
  # returned.
  #
  # ```
  # t = Tensor.from_array([[0, 1], [1, 1]]).astype(Float32)
  # puts t.eigvals
  #
  # # Tensor([-0.618,  1.618])
  # ```
  def eigvals
    assert_square_matrix(self)
    a = dup('F')
    n = a.shape[0]
    wr = Tensor(T).new([n])
    wl = wr.dup
    vl = Tensor(T).new([n, n])
    vr = wr.dup
    lapack(geev, 'N'.ord.to_u8, 'N'.ord.to_u8, n, a.to_unsafe, n, wr.to_unsafe,
      wl.to_unsafe, vl.to_unsafe, n, vr.to_unsafe, n, worksize: 3 * n)
    wr
  end

  # Matrix or vector norm.
  #
  # This function is able to return one of eight different matrix norms,
  # or one of an infinite number of vector norms (described below),
  # depending on the value of the ord parameter.
  #
  # ```crystal
  # t = Tensor.from_array([[0, 1], [1, 1], [1, 1], [2, 1]]).astype(Float32)
  # puts t.norm # => 3.6055512
  # ```
  def norm(*, order = 'F')
    assert_matrix(self)
    a = dup('F')
    m = a.shape[0]
    worksize = order == 'I' ? m : 0
    lapack_util(lange, worksize, order.ord.to_u8, m, m, tensor(a.to_unsafe), m)
  end

  # Compute the determinant of an array.
  #
  # ```crystal
  # t = Tensor.from_array([[1, 2], [3, 4]]).astype(Float32)
  # puts t.det # => -2.0
  # ```
  def det
    assert_square_matrix(self)
    a = dup('F')
    m, n = a.shape
    ipiv = Pointer(Int32).malloc(n)

    lapack(getrf, m, n, a.to_unsafe, n, ipiv)
    ldet = Statistics.prod(a.diag_view)
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
  # ```crystal
  # t = Tensor.from_array([[1, 2], [3, 4]]).astype(Float32)
  # puts t.inv
  #
  # # Tensor([[-2.0,  1.0],
  # #         [ 1.5, -0.5]])
  # ```
  def inv
    assert_square_matrix(self)
    a = dup('F')
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
  # ```crystal
  # a = Tensor.from_array([[3, 1], [1, 2]]).astype(Float32)
  # b = Tensor.from_array([9, 8]).astype(Float32)
  # puts a.solve(b)
  #
  # # Tensor([ 2.0,  3.0])
  # ```
  def solve(x : Tensor(T))
    assert_square_matrix(self)
    a = dup('F')
    x = x.dup('F')
    n = a.shape[0]
    m = x.ndims > 1 ? x.shape[1] : x.shape[0]
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
  def hessenberg
    assert_square_matrix(self)
    a = dup('F')

    if a.shape[0] < 2
      return a
    end

    n = a.shape[0]
    s = of_real_type(n)
    ilo = 0
    ihi = 0
    lapack(gebal, 'B'.ord.to_u8, n, a.to_unsafe, n, ilo, ihi, s.to_unsafe)
    tau = Tensor(T).new([n])
    lapack(gehrd, n, ilo, ihi, a.buffer, n, tau.buffer)
    Creation.triu(a, -1)
  end

  def matmul(other : Tensor(T))
    dest = Tensor(T).new([shape[0], other.shape[1]])
    no = LibCblas::CblasTranspose::CblasNoTrans
    blas(ge, mm, no, no, shape[0], other.shape[1], shape[1], 1.0, buffer, shape[0], other.buffer, other.shape[0], 1.0, dest.buffer, dest.shape[0])
    dest
  end
end
