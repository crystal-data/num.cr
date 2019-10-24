require "./object"
require "../api/math"
require "../api/stats"
require "../api/vectorprint"

class Bottle::Matrix(T) < Bottle::Internal::BottleObject(T)
  # Crystal slice pointing to the start of the matrix’s data.
  getter data : Slice(T)

  # Number of rows in the Matrix
  getter nrows : Int32

  # Number of columns in the Matrix
  getter ncols : Int32

  # Flag indicating if a Matrix is a view of another `Matrix`
  getter owner : Bool

  # Offset between rows in a Matrix
  getter tda : Int32

  # Creates a new `Matrix` of an arbitrary size from a given
  # indexable *data*.  The type of the Vector is inferred from
  # the provided data, as are the size, and stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # m         # => Matrix[[1, 2, 3], [4, 5, 6]]
  # typeof(v) # => Matrix(Int32)
  # ```
  def initialize(data : Indexable(Indexable(T)))
    check_type
    @nrows = data.size
    @ncols = data[0].size
    @tda = ncols
    @owner = true
    @data = Slice(T).new(nrows * ncols) do |idx|
      i = idx // ncols
      j = idx % ncols
      data[i][j]
    end
  end

  # Creates a new `Matrix` from a slice and strides.  This is
  # a lower level method that won't be used often by users,
  # but is very handy in cloning matrices, and creating matrices
  # from low level C libraries.
  #
  # ```
  # s = Slice.new(6) { |i| i + 1 }
  # m = Matrix.new s, 2, 3, 3, true
  #
  # m         # => Matrix[[1, 2, 3], [4, 5, 6]]
  # typeof(m) # => Matrix(Int32)
  # ```
  def initialize(@data : Slice(T), @nrows, @ncols, @tda, @owner)
    check_type
  end

  # Creates a new `Vector` from a block.  Infers the type
  # of the Vector from the value yielded by the block.
  #
  # ```
  # m = Matrix.new(2, 2) { |i, j| i + j }
  #
  # m         # => Matrix [[0, 1], [1, 2]]
  # typeof(m) # => Matrix(Int32)
  # ```
  def self.new(nrows : Int32, ncols : Int32, &block : Int32, Int32 -> T)
    data = Slice(T).new(nrows * ncols) do |idx|
      i = idx // ncols
      j = idx % ncols
      yield i, j
    end
    new(data, nrows, ncols, ncols, true)
  end

  # Selects a single value from a matrix. Calculates the offset
  # of the element from a given index, and the stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # get(0, 0) # => 1
  # ```
  private def get(i, j)
    check_sign(i)
    @data[i * tda + j]
  end

  # Sets a single value of a matrix. Calculates the offset
  # of the element from a given index, and the stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # set(0, 0, 8)
  # m # => Matrix[[8, 2, 3], [4, 5, 6]]
  # ```
  private def set(i, j, x)
    check_sign(i)
    @data[i * tda + j] = T.new(x)
  end

  # Selects a slice from a Matrix. Calculates the offset
  # of the elements from a given index, and the stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # get_n(0, 1, 3) # => Slice[2, 3, 4]
  # ```
  private def get_n(i, j, n)
    check_sign(i)
    @data[i * tda + j, n]
  end

  # Sets multiple values of a Matrix. Calculates the offset
  # of the elements from a given index, and the stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # set_n([0, 1], [0, 1], [10, 10])
  # m # => Matrix[[1, 10, 3], [4, 10, 6]]
  # ```
  private def set_n(is, js, xs)
    check_indexer(is, js, xs)
    xs.each_with_index { |e, i| set(is[i], js[i], e) }
  end

  # Gets a single row of a matrix.  The returned vector
  # shares memory with the Matrix.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # get_row(0, 1...) # => Vector[ 2  3]
  # ```
  private def get_row(i, jslice)
    f, l = range_to_slice(jslice, ncols)
    Vector.new data[i * tda + f, l - f], 1, false
  end

  # Gets a single column of a matrix.  The returned vector
  # shares memory with the Matrix.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # get_col(..., 0) # => Vector[ 1  4  7]
  # ```
  private def get_col(islice, j)
    f, l = range_to_slice(islice, nrows)
    start = f * tda + j
    finish = (l - 1) * tda + (j + 1)
    Vector.new data[start, finish - start], tda, false
  end

  # Selects a submatrix of a Matrix from ranges. Calculates
  # the offset of the elements from a given index, and the stride
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # get_slice(...2, ...2) # => Matrix[[1, 2], [4, 5]]
  # ```
  private def get_slice(islice, jslice)
    i, ni = range_to_slice(islice, nrows)
    j, nj = range_to_slice(jslice, ncols)
    start = i * tda + j
    finish = (ni - i) * tda + (nj - j) + 1 - start
    Matrix.new(data[start, finish], ni - i, nj - j, tda, false)
  end

  # Selects a single value from a matrix. Calculates the offset
  # of the element from a given index, and the stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # m[0, 0] # => 1
  # ```
  def [](i : Int32, j : Int32)
    get(i, j)
  end

  # Sets a single value of a matrix. Calculates the offset
  # of the element from a given index, and the stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # m[0, 0] = 8
  # m # => Matrix[[8, 2, 3], [4, 5, 6]]
  # ```
  def []=(i : Int32, j : Int32, x : Number)
    set(i, j, x)
  end

  # Gets a single row of a matrix.  The returned vector
  # shares memory with the Matrix.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[0] # => Vector[ 1  2  3]
  # ```
  def [](i : Int32, j : Range(Int32?, Int32?) = ...)
    get_row(i, j)
  end

  # Gets a single column of a matrix.  The returned vector
  # shares memory with the Matrix.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[..., 0] # => Vector[ 1  4  7]
  # ```
  def [](i : Range(Int32?, Int32?), j : Int32)
    get_col(i, j)
  end

  # Selects multiple non-contiguous elements from a
  # Matrix.  This method returns a copy since the memory
  # is not aligned.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # m[[0, 1], [1, 0]] # => Vector[2, 4]
  # ```
  def [](is, js)
    check_indexer(is, js)
    Vector.new(is.size) { |i| get(is[i], js[i]) }
  end

  # Sets multiple values of a Matrix. Calculates the offset
  # of the elements from a given index, and the stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # m[[0, 1], [0, 1]] = [10, 10]
  # m # => Matrix[[1, 10, 3], [4, 10, 6]]
  # ```
  def []=(is, js, xs)
    set_n(is, xs)
  end

  # Selects a submatrix of a Matrix from ranges. Calculates
  # the offset of the elements from a given index, and the stride
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[...2, ...2] # => Matrix[[1, 2], [4, 5]]
  # ```
  def [](islice : Range(Int32?, Int32?), jslice : Range(Int32?, Int32?))
    get_slice(islice, jslice)
  end

  # Computes the string representation of a Matrix.  This currently
  # truncates long rows, but needs to truncate long columns as well
  #
  # ```
  # m = Matrix.new(5, 5) { |i, j| i / (j + 1) }
  # puts m # =>
  #
  # [     0.0     0.0     0.0     0.0     0.0]
  # [     1.0     0.5   0.333    0.25     0.2]
  # [     2.0     1.0   0.667     0.5     0.4]
  # [     3.0     1.5     1.0    0.75     0.6]
  # [     4.0     2.0   1.333     1.0     0.8]
  # ```
  def to_s(io)
    nrows.times do |i|
      io << "["
      B::Util.vector_print(io, self[i], prefix = "")
      io << "\n"
    end
  end

  # Lazily yields the index values of a Matrix.  This method
  # is used as the core iteration method for Matrices
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_index { |i, j| puts "#{i}_#{j}" }
  #
  # # 0_0
  # # 0_1
  # # 1_0
  # # 1_1
  # ```
  def each_index(*, all = false, &block)
    nrows.times do |i|
      ncols.times do |j|
        yield i, j
      end
    end
  end

  # Lazily yields the values of a Matrix
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each { |i| puts i }
  #
  # # 1
  # # 2
  # # 4
  # # 5
  # ```
  def each(*, all = false, &block : Int32, Int32 -> T)
    each_index(all: all) { |i, j| yield self[i, j] }
  end

  # Lazily yields the values of a Matrix and the respective
  # index values.
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_with_index { |e, i, j| puts "#{i}_#{j}_#{e}" }
  #
  # # 0_0_1
  # # 0_1_2
  # # 1_0_4
  # # 1_1_5
  # ```
  def each_with_index(*, all = false, &block)
    each_index(all: all) { |i, j| yield(self[i, j], i, j) }
  end

  # Lazily yields the rows of a Matrix as vector views
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_row { |row| puts row }
  #
  # # Vector[  1  2]
  # # Vector[  4  5]
  # ```
  def each_row(*, all = false, &block)
    nrows.times do |row|
      yield self[row]
    end
  end

  # Lazily yields the rows of a Matrix as vector views
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_row_index { |row, i| puts row, i }
  #
  # # Vector[  1  2], 0
  # # Vector[  4  5], 1
  # ```
  def each_row_index(*, all = false, &block)
    nrows.times do |row|
      yield self[row], row
    end
  end

  # Lazily yields the columns of a Matrix as vector views
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_col { |col| puts col }
  #
  # # Vector[  1  4]
  # # Vector[  2  5]
  # ```
  def each_col(*, all = false, &block)
    ncols.times do |col|
      yield self[..., col]
    end
  end

  # Lazily yields the columns of a Matrix as vector views
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_col { |col, i| puts col, i }
  #
  # # Vector[  1  4], 0
  # # Vector[  2  5], 1
  # ```
  def each_col_index(*, all = false, &block)
    ncols.times do |col|
      yield self[..., col], col
    end
  end

  # Reduces an operation for each row or column of a Matrix.
  # Used to provide reduction operations along axes such
  # as sum, mean, max.  The most common reductions have aliases
  # on the class, for example `m.argmax(0)` is the same as
  # `m.reduce(0, &.argmax)`, but the function is public to
  # allow for flexible operations.
  #
  # ```
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m.reduce(1, &.max) # => [3, 6, 9]
  # ```
  def reduce(axis : Int32, &block : Vector(T) -> U) forall U
    ary = Vector.empty(axis == 0 ? ncols : nrows, dtype: U)
    if axis == 0
      each_col_index do |e, i|
        ary[i] = yield e
      end
    else
      each_row_index do |e, i|
        ary[i] = yield e
      end
    end
    ary
  end

  # Initializes a Matrix with an uninitialized slice
  # of data.  This is just another alias for zeros,
  # since Crystal right now doesn't support slices
  # pointing to generic memory, but if Crystal does
  # support this down the road, this will change
  #
  # ```crystal
  # m = Matrix.empty(3, 3, dtype: Int32)
  # m # => Matrix [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  # ```
  def self.empty(rows : Int32, cols : Int32, dtype : U.class = Float64) forall U
    new(Slice(U).new(rows * cols), rows, cols, cols, true)
  end

  # Initializes a Matrix full of zeros.  Default
  # dtype is Float64, but other dtypes are supported.
  #
  # ```crystal
  # m = Matrix.zeros(2, 2)
  # m # => Matrix [[0.0, 0.0], [0.0, 0.0]]
  # ```
  def self.zeros(rows : Int32, cols : Int32, dtype : U.class = Float64) forall U
    new(Slice(U).new(rows * cols), rows, cols, cols, true)
  end

  # Initializes a Matrix full of ones.  Default
  # dtype is Float64, but other dtypes are supported.
  #
  # ```crystal
  # m = Matrix.ones(2, 2)
  # m # => Matrix [[1.0, 1.0], [1.0, 1.0]]
  # ```
  def self.ones(rows : Int32, cols : Int32, dtype : U.class = Float64) forall U
    new(Slice(U).new(rows * cols, U.new(1)), rows, cols, cols, true)
  end

  # Initializes a Matrix full of a scalar.  Default
  # dtype is Float64, but other dtypes are supported.
  #
  # ```crystal
  # m = Matrix.full(2, 2, 4)
  # m # => Matrix [[4.0, 4.0], [4.0, 4.0]]
  # ```
  def self.full(rows : Int32, cols : Int32, x : Number, dtype : U.class = Float64) forall U
    new(Slice(U).new(rows * cols, U.new(x)), rows, cols, cols, true)
  end

  # Allocates a matrix full of random data
  # The dtype of the Matrix is inferred
  # from the values on either end of the
  # range.
  #
  # ```crystal
  # f = Matrix.random(0...10, 3, 3)
  # f # => Matrix [[2, 7, 8], [4, 9, 7], [6, 1, 5]]
  # ```
  def self.random(r : Range(U, U), rows : Int32, cols : Int32) forall U
    new(rows, cols) { |i, j| Random.rand(r) }
  end

  # Returns a copy of a Matrix that owns its own memory
  #
  # ```
  # f = Matrix.new [[1, 2], [3, 4]]
  # f.clone # => Vector[1, 2, 3, 4, 5]
  # ```
  def clone
    Matrix.new(data.dup, nrows, ncols, tda, true)
  end

  # Returns a flattened version of a Matrix, returns
  # a view when possible.
  #
  # ```
  # j = Matrix.new [[1, 2, 3], [4, 5, 6]]
  # j.ravel # => [1, 2, 3, 4, 5, 6]
  # j.owner # => false
  # ```
  def ravel
    if tda == ncols
      return Vector.new data, 1, false
    else
      m = Matrix(T).new(nrows, ncols) do |i, j|
        self[i, j]
      end
      return Vector.new m.data, 1, true
    end
  end

  # Casts a Matrix to another data dtype.
  # If the Matrix is already the given dtype,
  # a copy is not made, otherwise a new matrix
  # is returned.
  #
  # ```crystal
  # f = Matrix.new [[1, 2], [3, 4]]
  # f.astype(Float64) # => Matrix [[1.0, 2.0], [3.0, 4.0]]
  # ```
  def astype(dtype : U.class) forall U
    if T == U
      return self
    end
    Matrix.new(nrows, ncols) { |i, j| U.new(get(i, j)) }
  end

  # Elementwise addition of a matrix to another equally sized Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # add(j1, j1) # => [[2, 4], [6, 8]]
  # ```
  def +(other : Matrix)
    B.add(self, other)
  end

  # Elementwise addition of a Matrix to a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # add(j1, 2) # => [[3, 4], [5, 6]]
  # ```
  def +(other : Number)
    B.add(self, other)
  end

  # Elementwise subtraction of a Matrix to another equally sized Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # subtract(j1, j1) # => [[0, 0], [0, 0]]
  # ```
  def -(other : Matrix)
    B.subtract(self, other)
  end

  # Elementwise subtraction of a Matrix with a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # subtract(j1, 2) # => [[-1, 0], [1, 2]]
  # ```
  def -(other : Number)
    B.subtract(self, other)
  end

  # Elementwise multiplication of a Matrix to another equally sized Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # multiply(j1, j1) # => [[1, 4], [9, 16]]
  # ```
  def *(other : Matrix)
    B.multiply(self, other)
  end

  # Elementwise multiplication of a Matrix to a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # multiply(j1, 2) # => [[2, 4], [6, 8]]
  # ```
  def *(other : Number)
    B.multiply(self, other)
  end

  # Elementwise division of a Matrix to another equally sized Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # divide(j1, j2) # => [[1, 1], [1, 1]]
  # ```
  def /(other : Matrix)
    B.div(self, other)
  end

  # Elementwise division of a Matrix to a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # divide(j1, 2) # => [[0.5, 1.0], [1.5, 2.0]]
  # ```
  def /(other : Number)
    B.div(self, other)
  end

  # Returns the sum of a Matrix, or reduces
  # the operation along an axis.
  #
  # ```crystal
  # m = Matrix.new [[3, 3, 8], [0, 8, 3], [8, 9, 0]]
  # m.sum    # => 42
  # m.sum(0) # => Vector[  11  20  11]
  # m.sum(1) # => Vector[  14  11  17]
  # ```
  def sum(axis : Int32? = nil)
    if axis.nil?
      return self.ravel.sum
    end
    reduce(axis, &.sum)
  end

  # Returns the maximum value of a Matrix, or reduces
  # the operation along an axis.
  #
  # ```crystal
  # m = Matrix.new [[3, 3, 8], [0, 8, 3], [8, 9, 0]]
  # m.max    # => 9
  # m.max(0) # => Vector[  8  9  8]
  # m.max(1) # => Vector[  8  8  9]
  # ```
  def max(axis : Int32? = nil)
    if axis.nil?
      return self.ravel.max
    end
    reduce(axis, &.max)
  end

  # Returns the maximum index value of a Matrix, or reduces
  # the operation along an axis.
  #
  # ```crystal
  # m = Matrix.new [[3, 3, 8], [0, 8, 3], [8, 9, 0]]
  # m.argmax    # => 7
  # m.argmax(0) # => Vector[  2  2  0]
  # m.argmax(1) # => Vector[  2  1  1]
  # ```
  def argmax(axis : Int32? = nil)
    if axis.nil?
      return self.ravel.argmax
    end
    reduce(axis, &.argmax)
  end

  # Returns the minimum value of a Matrix, or reduces
  # the operation along an axis.
  #
  # ```crystal
  # m = Matrix.new [[3, 3, 8], [0, 8, 3], [8, 9, 0]]
  # m.min    # => 0
  # m.min(0) # => Vector[  0  3  0]
  # m.min(1) # => Vector[  3  0  0]
  # ```
  def min(axis : Int32? = nil)
    if axis.nil?
      return self.ravel.min
    end
    reduce(axis, &.min)
  end

  # Returns the minimum index value of a Matrix, or reduces
  # the operation along an axis.
  #
  # ```crystal
  # m = Matrix.new [[3, 3, 8], [0, 8, 3], [8, 9, 0]]
  # m.argmin    # => 3
  # m.argmin(0) # => Vector[  1  0  2]
  # m.argmin(1) # => Vector[  0  0  2]
  # ```
  def argmin(axis : Int32? = nil)
    if axis.nil?
      return self.ravel.argmin
    end
    reduce(axis, &.argmin)
  end
end
