require "../llib/lib_gsl"

class Matrix(T)
  @ptr : Pointer(LibGsl::GslMatrix)
  @nrows : Int32
  @ncols : Int32
  @mat : LibGsl::GslMatrix

  def initialize(data : Indexable(Indexable(T)))
    @nrows = data.size
    @ncols = data[0].size
    ptm = LibGsl.gsl_matrix_alloc(@nrows, @ncols)
    @ptr = ptm
    @mat = @ptr.value
  end

  def initialize(mat : LibGsl::GslMatrix, @nrows, @ncols)

  end
end

arr = [[1, 2], [3, 4]]
m = Matrix.new arr
