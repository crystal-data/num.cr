require "../core/matrix/*"
require "./*"

class Matrix(T, D)
  # Computes the maximum value of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # v.max # => 4
  # v.max(0) # => [3, 6]
  # ```
  def max(axis : Int32 | Nil = nil)
    Bottle::Core::MatrixStats.matrix_max(self, axis)
  end

  # Computes the minimum value of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # v.min # => 1
  # v.min(0) # => [1, 4]
  # ```
  def min(axis : Int32 | Nil = nil)
    Bottle::Core::MatrixStats.matrix_min(self, axis)
  end

  # Computes the min and max values of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3, 4]]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    Bottle::Core::MatrixStats.matrix_ptpv(self)
  end

  # Computes the "peak to peak" of a Matrix (max - min)
  #
  # ```
  # v = Matrix.new [[1, 2, 3, 4]]
  # v.ptp # => 3
  # ```
  def ptp
    Bottle::Core::MatrixStats.matrix_ptp(self)
  end

  # Computes the index of the maximum value of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # v.idxmax # => 5
  # v.idxmax(0) # => [2, 2]
  # ```
  def idxmax(axis : Int32 | Nil = nil)
    Bottle::Core::MatrixStats.matrix_idxmax(self, axis)
  end

  # Computes the index of the minimum value of a Matrix
  #
  # ```
  # v = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # v.idxmin # => 0
  # v.idxmin(1) # => [0, 0, 0]
  # ```
  def idxmin(axis : Int32 | Nil = nil)
    Bottle::Core::MatrixStats.matrix_idxmin(self, axis)
  end
end
