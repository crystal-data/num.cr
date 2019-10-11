require "../llib/lib_gsl"
require "./vector"
require "../util/indexing"
require "../util/arithmetic"
require "../util/statistics"

class Matrix(T)
  include Bottle::Util::Indexing
  include Bottle::Util::Arithmetic
  include Bottle::Util::Statistics

  @ptr : Pointer(LibGsl::GslMatrix)
  @nrows : Int32
  @ncols : Int32
  @mat : LibGsl::GslMatrix

  getter ptr
  getter nrows
  getter ncols

  def initialize(data : Indexable(Indexable(T)))
    @nrows = data.size
    @ncols = data[0].size
    ptm = LibGsl.gsl_matrix_alloc(@nrows, @ncols)
    data.each_with_index do |row, i|
      row.each_with_index do |col, j|
        LibGsl.gsl_matrix_set(ptm, i, j, col)
      end
    end
    @ptr = ptm
    @mat = @ptr.value
  end

  def initialize(mat : LibGsl::GslMatrix, @nrows, @ncols)
    @mat = mat
    @ptr = pointerof(@mat)
  end

  def copy
    ptm = LibGsl.gsl_matrix_alloc(@nrows, @ncols)
    LibGsl.gsl_matrix_memcpy(ptm, @ptr)
    return Matrix(Float64).new(ptm.value, @nrows, @ncols)
  end

  def +(other : Matrix)
    _add_mat(self.copy, other)
  end

  def +(other : Number)
    _add_mat_constant(self.copy, other)
  end

  def -(other : Matrix)
    _sub_mat(self.copy, other)
  end

  def -(other : Number)
    _sub_mat_constant(self.copy, other)
  end

  def *(other : Matrix)
    _mul_mat(self.copy, other)
  end

  def *(other : Number)
    _mul_mat_constant(self.copy, other)
  end

  def /(other : Matrix)
    _div_mat(self.copy, other)
  end

  def /(other : Number)
    _div_mat_constant(self.copy, other)
  end

  def max
    _mat_max(@ptr)
  end

  def min
    _mat_min(@ptr)
  end

  def ptpv
    _mat_ptpv(@ptr)
  end

  def ptp
    _mat_ptp(@ptr)
  end

  def idxmax
    _mat_idxmax(@ptr)
  end

  def idxmin
    _mat_idxmin(@ptr)
  end

  def transpose
    _transpose_mat(@ptr, @nrows, @ncols)
  end

  def [](row : Int32)
    _get_vec_at_row(@ptr, row, @ncols)
  end

  def [](row : Range(Nil, Nil), column : Int32)
    _get_vec_at_col(@ptr, column, @nrows)
  end

  def [](row : Range, col : Int32)
    r = _normalize_range(row, @nrows)
    c = _normalize_range(..., @ncols)
    mat = _slice_matrix_submatrix(@ptr, r.begin, r.end, c.begin, c.end)
    puts mat
    _get_vec_at_col(mat.ptr, col, mat.nrows)
  end

  def [](row : Range = ..., col : Range = ...)
    r = _normalize_range(row, @nrows)
    c = _normalize_range(col, @ncols)
    _slice_matrix_submatrix(@ptr, r.begin, r.end, c.begin, c.end)
  end

  def to_s(io)
    io << "["
    (0...@nrows).each do |el|
      startl = el == 0 ? "" : " "
      endl = el == @nrows - 1 ? "" : "\n"
      row = _get_vec_at_row(@ptr, el, @ncols)
      io << startl << row << endl
    end
    io << "]"
  end
end

def rand_matrix(n, m)
  return (0...n).map do |_|
    (0...m).map { |_| Random.rand(10) }
  end
end

m = Matrix.new rand_matrix(5, 5)
puts m
m[..., 2]
