require "../llib/gsl"
require "../util/indexing"
require "../util/arithmetic"
require "../util/statistics"
require "../util/generic"
require "../util/blas_arithmetic"

class Matrix(T, D)
  @ptr : Pointer(T)
  @obj : T
  @owner : Int32
  @tda : UInt64
  @nrows : UInt64
  @ncols : UInt64
  @data : Pointer(D)
  @dtype : D.class

  getter ptr
  getter owner
  getter tda
  getter nrows
  getter ncols
  getter dtype

  def self.new(data : Array(Array(A))) forall A
    new(fetch_struct(data))
  end

  def self.fetch_struct(data : Array(Array(Number)))
    nrows = data.size
    ncols = data[0].size
    ptm = LibGsl.gsl_matrix_alloc(nrows, ncols)
    data.each_with_index do |row, i|
      row.each_with_index do |col, j|
        LibGsl.gsl_matrix_set(ptm, i, j, col)
      end
    end
    return ptm, ptm.value.data
  end

  private def initialize(@ptr : Pointer(T), @data : Pointer(D))
    @obj = @ptr.value
    @owner = @obj.owner
    @tda = @obj.tda
    @nrows = @obj.size1
    @ncols = @obj.size2
    @dtype = D
  end

  def [](row : Int32)
    Bottle::Util::Indexing.get_matrix_row_at_index(@ptr, row)
  end

  def [](range : Range(Nil, Nil), column : Int32)
    Bottle::Util::Indexing.get_matrix_col_at_index(@ptr, column)
  end
end
