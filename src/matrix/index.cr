require "../core/matrix/*"
require "./*"

class Matrix(T, D)
  # Gets a single vector row view from a matrix at a given index
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[0] # => [1, 2, 3]
  # ```
  def [](row : Int32, range : Range = ...)
    Bottle::Core::MatrixIndex.get_matrix_row_at_index(self, row, range)
  end

  # Gets a single vector column view from a matrix at a given index
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[..., 0] # => [1, 4, 7]
  # ```
  def [](range : Range, column : Int32)
    Bottle::Core::MatrixIndex.get_matrix_col_at_index(self, column, range)
  end

  # Gets a submatrix from ranges
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[...2, ...2] # => [[1.0, 2.0], [4.0, 5.0]]
  # ```
  def [](rows : Range = ..., cols : Range = ...)
    Bottle::Core::MatrixIndex.get_submatrix_by_slice(self, rows, cols)
  end

  # Gets a single value from a matrix
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[2, 2] # => 9
  # ```
  def [](row : Int32, col : Int32)
    Bottle::Core::MatrixIndex.get_matrix_value(self, row, col)
  end
end
