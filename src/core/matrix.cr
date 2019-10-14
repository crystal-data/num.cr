require "../llib/gsl"
require "../util/indexing"
require "../util/arithmetic"
require "../util/statistics"
require "../util/generic"
require "../util/blas_arithmetic"

class Matrix(T)
  @ptr : Pointer(T)
  @obj : T
  @owner : Int32
  @tda : UInt64
  @nrows : UInt64
  @ncols : UInt64

  getter ptr
  getter owner
  getter tda
  getter nrows
  getter ncols

  def self.new(data : Array(Array(A))) forall A
    new(fetch_struct(data))
  end

  def self.fetch_struct(data : Array(Array(Float64)))
    nrows = data.size
    ncols = data[0].size
    ptm = LibGsl.gsl_matrix_alloc(nrows, ncols)
    data.each_with_index do |row, i|
      row.each_with_index do |col, j|
        LibGsl.gsl_matrix_set(ptm, i, j, col)
      end
    end
    return ptm
  end

  def self.fetch_struct(data : Array(Array(Int32)))
    nrows = data.size
    ncols = data[0].size
    ptm = LibGsl.gsl_matrix_int_alloc(nrows, ncols)
    data.each_with_index do |row, i|
      row.each_with_index do |col, j|
        LibGsl.gsl_matrix_int_set(ptm, i, j, col)
      end
    end
    return ptm
  end

  private def initialize(@ptr : Pointer(T))
    @obj = @ptr.value
    @owner = @obj.owner
    @tda = @obj.tda
    @nrows = @obj.size1
    @ncols = @obj.size2
  end

  def [](row : Int32)
    Bottle::Util::Indexing.get_matrix_row_at_index(@ptr, row)
  end

  def [](range : Range(Nil, Nil), column : Int32)
    Bottle::Util::Indexing.get_matrix_col_at_index(@ptr, column)
  end
end
