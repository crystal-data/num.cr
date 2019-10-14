macro matrix_indexing_abstract(type_, dtype, prefix)
  module Bottle::Util::Indexing
    include Bottle::Util::Exceptions
    extend self

    def get_matrix_row_at_index(matrix : Pointer({{ type_ }}), index : Int32)
      vv = LibGsl.{{ prefix }}_row(matrix, index)
      return Vector.new vv.vector, vv.vector.data
    end

    def get_matrix_col_at_index(matrix : Pointer({{ type_ }}), column : Int32)
      vv = LibGsl.{{ prefix }}_column(matrix, column)
      return Vector.new vv.vector, vv.vector.data
    end
  end
end

matrix_indexing_abstract LibGsl::GslMatrix, Float64, gsl_matrix
matrix_indexing_abstract LibGsl::GslMatrixInt, Int32, gsl_matrix_int
