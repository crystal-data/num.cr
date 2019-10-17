require "../../matrix/*"
require "../../vector/*"
require "../../libs/gsl"
require "../../libs/cblas"

module Bottle::Core
  extend self

  # Outer product of two vectors
  #
  # ```
  # v = Vector.new [1, 2, 3]
  # Bottle.outer(v, v)
  # [[1.0, 2.0, 3.0]
  #  [2.0, 4.0, 6.0]
  #  [3.0, 6.0, 9.0]]
  # ```
  def outer(x : Vector, y : Vector)
    m = Matrix.empty(x.size, y.size)
    LibCblas.dger(
      LibCblas::MatrixLayout::RowMajor,
      m.nrows,
      m.ncols,
      1.0,
      x.data,
      x.stride,
      y.data,
      y.stride,
      m.data,
      m.tda,
    )
    return m
  end

  def givens(x : Vector, y : Vector, da : Float64, db : Float64, c : Float64, s : Float64)
    dx = x.copy
    dy = y.copy
    LibCblas.drotg(da, db, pointerof(c), pointerof(s))
    LibCblas.drot(dx.size, dx.data, dx.stride, dy.data, dy.stride, c, s)
    return dx, dy
  end
end
