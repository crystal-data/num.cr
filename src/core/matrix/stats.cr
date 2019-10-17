require "../../libs/gsl"
require "../../matrix/*"

module Bottle::Core::MatrixStats
  extend self

  # Computes the maximum value of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # v.max # => 4
  # v.max(0) # => [3, 6]
  # ```
  def matrix_max(m : Matrix(LibGsl::GslMatrix, Float64), axis : Int32 | Nil = nil)
    if axis.nil?
      return LibGsl.gsl_matrix_max(m.ptr)
    end
    if axis == 0
      v = Vector.empty(m.nrows)
      (0...m.nrows).each do |i|
        v[i] = m[i].max
      end
      return v
    end
    if axis == 1
      v = Vector.empty(m.ncols)
      (0...m.ncols).each do |i|
        v[i] = m[..., i].max
      end
      return v
    end
  end

  # Computes the minimum value of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # v.min # => 1
  # v.min(0) # => [1, 4]
  # ```
  def matrix_min(m : Matrix(LibGsl::GslMatrix, Float64), axis : Int32 | Nil = nil)
    if axis.nil?
      return LibGsl.gsl_matrix_min(m.ptr)
    end
    if axis == 0
      v = Vector.empty(m.nrows)
      (0...m.nrows).each do |i|
        v[i] = m[i].min
      end
      return v
    end
    if axis == 1
      v = Vector.empty(m.ncols)
      (0...m.ncols).each do |i|
        v[i] = m[..., i].min
      end
      return v
    end
  end

  # Computes the min and max values of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3, 4]]
  # v.ptpv # => {1, 4}
  # ```
  def matrix_ptpv(m : Matrix(LibGsl::GslMatrix, Float64))
    min_out = 0.0
    max_out = 0.0
    LibGsl.gsl_matrix_minmax(m.ptr, pointerof(min_out), pointerof(max_out))
    return min_out, max_out
  end

  # Computes the "peak to peak" of a Matrix (max - min)
  #
  # ```
  # v = Matrix.new [[1, 2, 3, 4]]
  # v.ptp # => 3
  # ```
  def matrix_ptp(m : Matrix(LibGsl::GslMatrix, Float64))
    mn, mx = matrix.ptpv(m)
    return mx - mn
  end

  # Computes the index of the maximum value of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # v.idxmax # => 5
  # v.idxmax(0) # => [2, 2]
  # ```
  def matrix_idxmax(m : Matrix(LibGsl::GslMatrix, Float64), axis : Int32 | Nil = nil)
    if axis.nil?
      imin = UInt64.new 0
      imax = UInt64.new 0
      LibGsl.gsl_matrix_max_index(m.ptr, pointerof(imin), pointerof(imax))
      return imin * m.ncols + imax
    end
    if axis == 0
      v = Vector.empty(m.nrows)
      (0...m.nrows).each do |i|
        v[i] = m[i].idxmax
      end
      return v
    end
    if axis == 1
      v = Vector.empty(m.ncols)
      (0...m.ncols).each do |i|
        v[i] = m[..., i].idxmax
      end
      return v
    end
  end

  # Computes the index of the minimum value of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # v.idxmin # => 0
  # v.idxmin(1) # => [0, 0, 0]
  # ```
  def matrix_idxmin(m : Matrix(LibGsl::GslMatrix, Float64), axis : Int32 | Nil = nil)
    if axis.nil?
      imin = UInt64.new 0
      imax = UInt64.new 0
      return LibGsl.gsl_matrix_min_index(m.ptr, pointerof(imin), pointerof(imax))
    end
    if axis == 0
      v = Vector.empty(m.nrows)
      (0...m.nrows).each do |i|
        v[i] = m[i].idxmin
      end
      return v
    end
    if axis == 1
      v = Vector.empty(m.ncols)
      (0...m.ncols).each do |i|
        v[i] = m[..., i].idxmin
      end
      return v
    end
  end
end
