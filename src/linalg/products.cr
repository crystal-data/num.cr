require "../tensor/tensor"
require "../libs/lapack"
require "../libs/cblas"

module Bottle::LinAlg
  private def vector_shape_match(a, b)
    if a.shape != b.shape
      raise Internal::Exceptions::ShapeError.new("Shapes do not match")
    end
  end

  private def matrix_shape_match(a, b)
    if a.shape[-1] != b.shape[-2]
      raise Internal::Exceptions::ShapeError.new("Matrices are not compatible")
    end
  end

  private def insist_1d(a, b)
    if a.ndims != 1 || b.ndims != 1
      raise Internal::Exceptions::ShapeError.new("Inputs must be 1D")
    end
  end

  macro linalg(dtype, prefix)

    def dot(a : Tensor({{dtype}}), b : Tensor({{dtype}}))
      aptr = a.buffer
      bptr = b.buffer

      if a.ndims == 1 && b.ndims == 1
        vector_shape_match(a, b)
        dp = LibCblas.{{prefix}}dot(
          a.size,
          aptr,
          a.strides[0],
          bptr,
          b.strides[0],
        )

        Tensor({{dtype}}).new([1]) { |_| dp }

      elsif a.ndims == 2 && b.ndims == 2
        if a.flags.fortran?
          a = a.dup
        end

        if b.flags.fortran?
          b = b.dup
        end

        dest = Tensor({{dtype}}).new([a.shape[0], b.shape[1]])

        LibCblas.{{prefix}}gemm(
          LibCblas::MatrixLayout::RowMajor,
          LibCblas::MatrixTranspose::NoTrans,
          LibCblas::MatrixTranspose::NoTrans,
          a.shape[0],
          b.shape[1],
          a.shape[1],
          {{dtype}}.new(1),
          aptr,
          a.strides[0],
          bptr,
          b.strides[0],
          {{dtype}}.new(0),
          dest.buffer,
          dest.strides[0],
        )
        dest
      else
        raise "Higher dimension matrix multiplication is not supported"
      end
    end

    def inner(a : Tensor({{dtype}}), b : Tensor({{dtype}}))
      insist_1d(a, b)
      dot(a, b)
    end

    def outer(a : Tensor({{dtype}}), b : Tensor({{dtype}}))
      insist_1d(a, b)

      m = Tensor({{dtype}}).new([a.size, b.size])

      LibCblas.{{prefix}}ger(
        LibCblas::MatrixLayout::RowMajor,
        a.size,
        b.size,
        {{dtype}}.new(1),
        a.buffer,
        a.strides[0],
        b.buffer,
        b.strides[0],
        m.buffer,
        m.strides[0],
      )
      m
    end

  end

  linalg(Float64, d)
  linalg(Float32, s)
end
