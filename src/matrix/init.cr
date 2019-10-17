require "../core/matrix/*"
require "./*"

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
  getter data

  def copy
    Bottle::Core::MatrixIndex.copy_matrix(self)
  end

  def self.new(data : Array(Array(A))) forall A
    new(*fetch_struct(data))
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

  def initialize(@ptr : Pointer(T), @data : Pointer(D))
    @obj = @ptr.value
    @owner = @obj.owner
    @tda = @obj.tda
    @nrows = @obj.size1
    @ncols = @obj.size2
    @dtype = D
  end

  def initialize(@obj : T, @data : Pointer(D))
    @ptr = pointerof(@obj)
    @owner = @obj.owner
    @tda = @obj.tda
    @nrows = @obj.size1
    @ncols = @obj.size2
    @dtype = D
  end

  def to_s(io)
    io << "["
    (0...@nrows).each do |el|
      startl = el == 0 ? "" : " "
      endl = el == @nrows - 1 ? "" : "\n"
      row = Bottle::Core::MatrixIndex.get_matrix_row_at_index(self, el, ...)
      io << startl << row << endl
    end
    io << "]"
  end

  def self.empty(nrows, ncols)
    m = LibGsl.gsl_matrix_alloc(nrows, ncols)
    return Matrix.new m, m.value.data
  end

  def self.random(nrows, ncols)
    d = (0...nrows).map do |_|
      (0...ncols).map { |_| Random.rand }
    end
    return Matrix.new d
  end
end
