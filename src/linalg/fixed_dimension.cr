require "../core/ndtensor"
require "../libs/lapack"
require "../libs/cblas"

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
      if dx.shape != dy.shape
        raise "Shapes #{dx.shape} and #{dy.shape} are not aligned"
      end

      if dx.ndims > 1
        raise "Only one-dimension tensors are supported"
      end

      LibCblas.{{prefix}}dot(
        dx.size,
        dx.@buffer,
        dx.strides[0],
        dy.@buffer,
        dy.strides[0],
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
      if dx.shape != dy.shape
        raise "Shapes #{dx.shape} and #{dy.shape} are not aligned"
      end

      if dx.ndims > 1
        raise "Only one-dimension tensors are supported"
      end

      m = Tensor({{dtype}}).new([dx.shape[0], dy.shape[0]])

      LibCblas.{{prefix}}ger(
        LibCblas::MatrixLayout::RowMajor,
        dx.size,
        dy.size,
        {{dtype}}.new(1),
        dx.@buffer,
        dx.strides[0],
        dy.@buffer,
        dy.strides[0],
        m.@buffer,
        m.strides[0],
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
    def matmul(dx : Tensor({{dtype}}), dy : Tensor({{dtype}}))
      if dx.shape[1] != dy.shape[0]
        raise "Matrices cannot be multiplied together"
      end

      if dx.ndims > 2
        raise "Only two dimensional tensors are currently supported"
      end

      m = Tensor({{dtype}}).new([dx.shape[0], dy.shape[1]])

      LibCblas.{{prefix}}gemm(
        LibCblas::MatrixLayout::RowMajor,
        LibCblas::MatrixTranspose::NoTrans,
        LibCblas::MatrixTranspose::NoTrans,
        dx.shape[0],
        dy.shape[1],
        dx.shape[1],
        {{dtype}}.new(1),
        dx.@buffer,
        dx.strides[0],
        dy.@buffer,
        dy.strides[0],
        {{dtype}}.new(0),
        m.@buffer,
        m.strides[0],
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
      if x.ndims > 1
        raise "Only one-dimensional tensors are supported"
      end
      LibCblas.{{prefix}}nrm2(
        x.size,
        x.@buffer,
        x.strides[0],
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
    def inv(a : Tensor({{dtype}}))
      if a.shape[0] != a.shape[1]
        raise "Matrix must be square"
      end

      if a.ndims > 2
        raise "Only 2 dimensional square matrices are supported"
      end

      a = a.dup
      dim = a.shape[0]
      ipiv = Pointer(Int32).malloc(dim)

      m = a.shape[0]
      n = a.shape[1]
      lda = a.strides[0]

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
