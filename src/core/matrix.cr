require "./tensor"
require "../api/math"
require "../api/stats"
require "../api/vectorprint"

class Bottle::Matrix(T)
  # Number of rows in the Matrix
  getter nrows : Int32

  # Number of columns in the Matrix
  getter ncols : Int32

  @tda : Int32

  # Creates a new `Matrix` of an arbitrary size from a given
  # indexable *data*.  The type of the Matrix is inferred from
  # the provided data, as are the size, and stride.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # m         # => Matrix[[1, 2, 3], [4, 5, 6]]
  # typeof(v) # => Matrix(Int32)
  # ```
  def initialize(data : Indexable(Indexable(T)))
    @nrows = data.size
    @ncols = data[0].size
    @tda = ncols
    @owner = true
    @buffer = Pointer(T).malloc(nrows * ncols) do |idx|
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
  def initialize(@buffer : Pointer(T), @nrows, @ncols, @tda, @owner)
  end

  # Creates a new `Matrix` from a block.  Infers the type
  # of the Tensor from the value yielded by the block.
  #
  # ```
  # m = Matrix.new(2, 2) { |i, j| i + j }
  #
  # m         # => Matrix [[0, 1], [1, 2]]
  # typeof(m) # => Matrix(Int32)
  # ```
  def self.new(nrows : Int32, ncols : Int32, &block : Int32, Int32 -> T)
    data = Pointer(T).malloc(nrows * ncols) do |idx|
      i = idx // ncols
      j = idx % ncols
      yield i, j
    end
    new(data, nrows, ncols, ncols, true)
  end

  private def check_index_out_of_bounds(i, j)
    check_index_out_of_bounds(i, j) { raise IndexError.new }
  end

  private def check_index_out_of_bounds(i, j)
    i += nrows if i < 0
    j += ncols if j < 0
    if (0 <= i < nrows) && (0 <= j < ncols)
      i * @tda + j
    else
      yield
    end
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
    index = check_index_out_of_bounds i, j
    @buffer[index]
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
    index = check_index_out_of_bounds i, j
    @buffer[index] = T.new(x)
  end

  # Gets a single row of a matrix.  The returned Tensor
  # shares memory with the Matrix.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[0] # => Tensor[ 1  2  3]
  # ```
  def [](i : Int32, j : Range(Int32?, Int32?) = ...)
    start, offset = Indexable.range_to_index_and_count(j, ncols)
    Tensor.new @buffer + (@tda * i) + start, offset, 1, false
  end

  # Gets a single column of a matrix.  The returned Tensor
  # shares memory with the Matrix.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[..., 0] # => Tensor[ 1  4  7]
  # ```
  def [](i : Range(Int32?, Int32?), j : Int32)
    start, offset = Indexable.range_to_index_and_count(i, nrows)
    Tensor.new @buffer + (@tda * start) + j, offset, @tda, false
  end

  # Selects a submatrix of a Matrix from ranges. Calculates
  # the offset of the elements from a given index, and the stride
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[...2, ...2] # => Matrix[[1, 2], [4, 5]]
  # ```
  def [](islice : Range(Int32?, Int32?), jslice : Range(Int32?, Int32?))
    istart, ioffset = Indexable.range_to_index_and_count(islice, nrows)
    jstart, joffset = Indexable.range_to_index_and_count(jslice, ncols)

    Matrix.new @buffer + (@tda * istart) + jstart, ioffset, joffset, @tda, false
  end

  def to_s(io)
    mx = max
    nrows.times do |i|
      prefix = i == 0 ? "Matrix[[" : "       ["
      B::Util.vector_print(io, self[i], prefix: prefix, override_max: true, maxval: mx)
      io << "]" unless i != nrows - 1
      io << "\n" unless i == nrows - 1
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
  def each_index(&block)
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
  def each(&block : Int32, Int32 -> T)
    each_index { |i, j| yield self[i, j] }
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
  def each_with_index(&block)
    each_index { |i, j| yield(self[i, j], i, j) }
  end

  # Lazily yields the rows of a Matrix as Tensor views
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_row_index { |row, i| puts row, i }
  #
  # # Tensor[  1  2], 0
  # # Tensor[  4  5], 1
  # ```
  def each_row_index(*, all = false, &block)
    nrows.times do |row|
      yield self[row], row
    end
  end

  # Lazily yields the columns of a Matrix as Tensor views
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_col { |col| puts col }
  #
  # # Tensor[  1  4]
  # # Tensor[  2  5]
  # ```
  def each_col(*, all = false, &block)
    ncols.times do |col|
      yield self[..., col]
    end
  end

  # Lazily yields the columns of a Matrix as Tensor views
  #
  # ```crystal
  # m = Matrix.new [[1, 2], [4, 5]]
  # m.each_col { |col, i| puts col, i }
  #
  # # Tensor[  1  4], 0
  # # Tensor[  2  5], 1
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
  def reduce(axis : Int32, &block : Tensor(T) -> U) forall U
    ary = Tensor.empty(axis == 0 ? ncols : nrows, dtype: U)
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
    new(Pointer(U).malloc(rows * cols), rows, cols, cols, true)
  end

  # Initializes a Matrix full of zeros.  Default
  # dtype is Float64, but other dtypes are supported.
  #
  # ```crystal
  # m = Matrix.zeros(2, 2)
  # m # => Matrix [[0.0, 0.0], [0.0, 0.0]]
  # ```
  def self.zeros(rows : Int32, cols : Int32, dtype : U.class = Float64) forall U
    new(Pointer(U).malloc(rows * cols), rows, cols, cols, true)
  end

  # Initializes a Matrix full of ones.  Default
  # dtype is Float64, but other dtypes are supported.
  #
  # ```crystal
  # m = Matrix.ones(2, 2)
  # m # => Matrix [[1.0, 1.0], [1.0, 1.0]]
  # ```
  def self.ones(rows : Int32, cols : Int32, dtype : U.class = Float64) forall U
    new(Pointer(U).malloc(rows * cols, U.new(1)), rows, cols, cols, true)
  end

  # Initializes a Matrix full of a scalar.  Default
  # dtype is Float64, but other dtypes are supported.
  #
  # ```crystal
  # m = Matrix.full(2, 2, 4)
  # m # => Matrix [[4.0, 4.0], [4.0, 4.0]]
  # ```
  def self.full(rows : Int32, cols : Int32, x : Number, dtype : U.class = Float64) forall U
    new(Pointer(U).malloc(rows * cols, U.new(x)), rows, cols, cols, true)
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
    new(rows, cols) { |_, _| Random.rand(r) }
  end

  def clone
    Matrix(T).new(nrows, ncols) { |i, j| @buffer[i * tda + j] }
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
    if @tda == ncols
      Tensor.new @buffer, nrows * ncols, 1, false
    else
      m = Matrix(T).new(nrows, ncols) do |i, j|
        self[i, j]
      end
      Tensor.new m.@buffer, nrows * ncols, 1, true
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
    B.divide(self, other)
  end

  # Elementwise division of a Matrix to a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # divide(j1, 2) # => [[0.5, 1.0], [1.5, 2.0]]
  # ```
  def /(other : Number)
    B.divide(self, other)
  end

  # Returns the sum of a Matrix, or reduces
  # the operation along an axis.
  #
  # ```crystal
  # m = Matrix.new [[3, 3, 8], [0, 8, 3], [8, 9, 0]]
  # m.sum    # => 42
  # m.sum(0) # => Tensor[  11  20  11]
  # m.sum(1) # => Tensor[  14  11  17]
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
  # m.max(0) # => Tensor[  8  9  8]
  # m.max(1) # => Tensor[  8  8  9]
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
  # m.argmax(0) # => Tensor[  2  2  0]
  # m.argmax(1) # => Tensor[  2  1  1]
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
  # m.min(0) # => Tensor[  0  3  0]
  # m.min(1) # => Tensor[  3  0  0]
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
  # m.argmin(0) # => Tensor[  1  0  2]
  # m.argmin(1) # => Tensor[  0  0  2]
  # ```
  def argmin(axis : Int32? = nil)
    if axis.nil?
      return self.ravel.argmin
    end
    reduce(axis, &.argmin)
  end
end
