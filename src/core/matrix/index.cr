require "../../libs/gsl"
require "../../matrix/*"
require "../vector/index"

module Bottle::Core::MatrixIndex
  include Bottle::Core::Exceptions
  extend self

  # Returns a copy of a matrix
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # n = m.copy
  # n # => [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # ```
  def copy_matrix(matrix : Matrix(LibGsl::GslMatrix, Float64))
    ptm = LibGsl.gsl_matrix_alloc(matrix.nrows, matrix.ncols)
    LibGsl.gsl_matrix_memcpy(ptm, matrix.ptr)
    return Matrix.new ptm, ptm.value.data
  end

  # Returns a single value from a matrix
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[0, 0] # => 1
  # ```
  def get_matrix_value(matrix : Matrix(LibGsl::GslMatrix, Float64), row, col)
    LibGsl.gsl_matrix_get(matrix.ptr, row, col)
  end

  # Gets row view from a matrix at a given index
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[0] # => [1, 2, 3]
  # ```
  def get_matrix_row_at_index(matrix : Matrix(LibGsl::GslMatrix, Float64), index : Int32, range)
    r = Bottle::Core::VectorIndex.convert_range_to_slice(range, matrix.ncols)
    vv = LibGsl.gsl_matrix_subrow(matrix.ptr, index, r.begin, r.end - r.begin)
    return Vector.new vv.vector, vv.vector.data
  end

  # Gets a column view from a matrix at a given index
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[..., 0] # => [1, 4, 7]
  # ```
  def get_matrix_col_at_index(matrix : Matrix(LibGsl::GslMatrix, Float64), column : Int32, range)
    c = Bottle::Core::VectorIndex.convert_range_to_slice(range, matrix.nrows)
    vv = LibGsl.gsl_matrix_subcolumn(matrix.ptr, column, c.begin, c.end - c.begin)
    return Vector.new vv.vector, vv.vector.data
  end

  # Gets a submatrix from ranges
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[...2, ...2] # => [[1.0, 2.0], [4.0, 5.0]]
  # ```
  def get_submatrix_by_slice(matrix : Matrix(LibGsl::GslMatrix, Float64), rows, cols)
    r = Bottle::Core::VectorIndex.convert_range_to_slice(rows, matrix.nrows)
    c = Bottle::Core::VectorIndex.convert_range_to_slice(cols, matrix.ncols)
    m = LibGsl.gsl_matrix_submatrix(matrix.ptr, r.begin, c.begin, c.end - c.begin, r.end - r.begin)
    return Matrix.new m.matrix, m.matrix.data
  end

  def set_matrix_row(matrix : Matrix(LibGsl::GslMatrix, Float64), row : Int32, vector : Vector(LibGsl::GslVector, Float64))
    LibGsl.gsl_matrix_set_row(matrix.ptr, row, vector.ptr)
  end

  def set_matrix_column(matrix : Matrix(LibGsl::GslMatrix, Float64), column : Int32, vector : Vector(LibGsl::GslVector, Float64))
    LibGsl.gsl_matrix_set_col(matrix.ptr, column, vector.ptr)
  end

  # Returns the diagonal of a matrix
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m.diagonal # => [1.0, 5.0, 9.0]
  # ```
  def get_matrix_diagonal(matrix : Matrix, k : Int32)
    vv = LibGsl.gsl_matrix_subdiagonal(matrix.ptr, k)
    return Vector.new vv.vector, vv.vector.data
  end
end
