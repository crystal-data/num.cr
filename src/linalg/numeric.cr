require "../core/tensor"
require "../libs/lapack"

module Bottle::Internal::LinAlg
  macro linalg(dtype, prefix)
    # Dot product of two `Tensor`s. Specifically, this is the inner
    # product of two `Tensor`s without the complex conjugate.
    #
    # ```
    # t = Tensor.new [1.0, 2.0, 3.0]
    #
    # dot(t, t) # => 14.0
    # ```
    def dot(dx : Tensor({{dtype}}), dy : Tensor({{dtype}}))
      if dx.size != dy.size
        raise "Shapes #{dx.size} and #{dy.size} are not aligned"
      end

      LibCblas.{{prefix}}dot(
        dx.size,
        dx.@buffer,
        dx.@stride,
        dy.@buffer,
        dy.@stride,
      )
    end

    # Inner product of two `Tensor`s without the complex conjugate.
    #
    # ```
    # t = Tensor.new [1.0, 2.0, 3.0]
    #
    # inner(t, t) # => 14.0
    # ```
    def inner(dx : Tensor({{dtype}}), dy : Tensor({{dtype}}))
      dot(dx, dy)
    end

    # Compute the outer product of two `Tensor`s.
    #
    # Given two vectors, `a = [a0, a1, ..., aM]` and
    # `b = [b0, b1, ..., bN]`, the outer product is:
    #
    # ```
    # # [[a0*b0  a0*b1 ... a0*bN ]
    # #  [a1*b0    .
    # #  [ ...          .
    # #  [aM*b0            aM*bN ]]
    # ```
    def outer(dx : Tensor({{dtype}}), dy : Tensor({{dtype}}))
      m = Matrix.empty(dx.size, dy.size)

      LibCblas.{{prefix}}ger(
        LibCblas::MatrixLayout::RowMajor,
        dx.size,
        dy.size,
        {{dtype}}.new(1.0),
        dx.@buffer,
        dx.@stride,
        dy.@buffer,
        dy.@stride,
        m.@buffer,
        m.@tda,
      )
      m
    end

    # Computes the Matrix product of two matrices.
    #
    # ```
    # m = Matrix.new [[1.0, 2.0], [3.0, 4.0]]
    # matmul(m, m)
    #
    # # Matrix[[      7.0     10.0]
    # #        [     15.0     22.0]]
    # ```
    def matmul(dx : Matrix({{dtype}}), dy : Matrix({{dtype}}))
      if dx.ncols != dy.nrows
        raise "Matrices cannot be multiplied together"
      end

      m = Matrix.empty(dx.nrows, dy.ncols)

      LibCblas.{{prefix}}gemm(
        LibCblas::MatrixLayout::RowMajor,
        LibCblas::MatrixTranspose::NoTrans,
        LibCblas::MatrixTranspose::NoTrans,
        dx.nrows,
        dy.ncols,
        dx.ncols,
        {{dtype}}.new(1),
        dx.@buffer,
        dx.@tda,
        dy.@buffer,
        dy.@tda,
        {{dtype}}.new(0),
        m.@buffer,
        m.@tda,
      )
      m
    end

    # Returns the euclidean norm of a vector via the function
    # name, so that
    #
    # norm(x) := sqrt( x'*x )
    #
    # ```
    # t = Tensor.new [1.0, 2.0, 3.0]
    #
    # norm(t) # => 3.7416573867739413
    # ```
    def norm(x : Tensor({{dtype}}))
      LibCblas.{{prefix}}nrm2(
        x.size,
        x.@buffer,
        x.@stride,
      )
    end

    # Compute the (multiplicative) inverse of a `Matrix.
    #
    # Given a square matrix a, return the matrix ainv satisfying
    # dot(a, ainv) = dot(ainv, a) = eye(a.shape[0]).
    #
    # ```
    # m = Matrix.new [[1.0, 2.0], [3.0, 4.0]]
    #
    # inv(m)
    #
    # # Matrix[[    -2.0     1.0]
    # #        [     1.5    -0.5]]
    # ```
    def inv(a : Matrix({{dtype}}))
      if a.nrows != a.ncols
        raise "Matrix must be square"
      end

      a = a.clone
      dim = Math.min(a.nrows, a.ncols)
      ipiv = Pointer(Int32).malloc(dim)

      m = a.nrows
      n = a.ncols
      lda = a.@tda

      LibLapack.{{prefix}}getrf(
        pointerof(m),
        pointerof(n),
        a.@buffer,
        pointerof(lda),
        ipiv,
        out info
      )

      order = dim
      lwork = m * n
      work = Pointer({{dtype}}).malloc(lwork)

      LibLapack.{{prefix}}getri(
        pointerof(order),
        a.@buffer,
        pointerof(lda),
        ipiv,
        work,
        pointerof(lwork),
        out invinfo,
      )

      return a
    end
  end

  linalg(Float64, d)
  linalg(Float32, s)
end
