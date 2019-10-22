require "./flask"
require "./indexing"
require "../libs/dtype"
require "../ma/mask"
require "./arithmetic"
require "../linalg/*"
require "../blas/*"
require "../ufunc/*"

class Jug(T)
  include Bottle::Internal::Indexing
  include Bottle::Internal::Dtype
  include Bottle

  getter data : Slice(T)
  getter nrows : Int32
  getter ncols : Int32
  getter tda : Int32

  # Converts a Jug into a string representation.  This currently
  # displays all the values of a Jug, but should re-worked to
  # truncate large Jugs
  #
  # ```
  # j = Jug.new [[1, 2], [3, 4]]
  # puts j # => [[1, 2], [3, 4]]
  # ```
  def to_s(io)
    io << "["
    (0...@nrows).each do |el|
      startl = el == 0 ? "" : " "
      endl = el == @nrows - 1 ? "" : "\n"
      row = self[el]
      io << startl << row << endl
    end
    io << "]"
  end

  # Returns a copy of a Jug that owns its own memory
  #
  # ```
  # j = Jug.new [[1, 2], [3, 4]]
  # j.clone # => [[1, 2], [3, 4]]
  # ```
  def clone
    Jug(T).new data.dup, nrows, ncols, tda
  end

  def self.new(nrows, ncols, &block)
    data = Slice(T).new(nrows * ncols) do |idx|
      i = idx // ncols
      j = idx % ncols
      yield i, j
    end
    new(data, nrows, ncols, ncols)
  end

  # Initializes a Jug from an Indexable of a type.
  # this is the common user facing init function, and
  # allocates a slice and sets its data to the elements
  # of data
  def initialize(data : Indexable(Indexable(T)))
    @nrows = data.size
    @ncols = data[0].size
    @tda = ncols
    @data = Slice(T).new(nrows * ncols) do |idx|
      i = idx // ncols
      j = idx % ncols
      data[i][j]
    end
  end

  # Primarily a convenience method to allow for cloning
  # of Vectors, should not be called by outside methods.
  def initialize(@data, @nrows, @ncols, @tda)
  end

  # Iterates over each index of a Jug.  The core operation
  # to provide support for reduction and accumulation of
  # jugs.
  def each_index(*, all = false, &block)
    nrows.times do |i|
      ncols.times do |j|
        yield i, j
      end
    end
  end

  # Reduces an operation for each row or column of a Jug.
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
  def reduce(axis : Indexer, &block : Flask(T) -> U) forall U
    ary = Flask.empty(axis == 0 ? ncols : nrows, dtype = U)
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

  # Iterates through each value of a Jug.  Iteration
  # is done in row-major order.
  def each(*, all = false, &block)
    each_index(all: all) { |i, j| yield self[i, j] }
  end

  # Iterates through each value and index value of a Jug.  Iteration
  # is done in row-major order.
  def each_with_index(*, all = false, &block)
    each_index(all: all) { |i, j| yield(self[i, j], i, j) }
  end

  # Iterates through each Flask view of a row in a Jug.
  # Iterates in increasing order, and returns views of
  # each Flask.
  def each_row(*, all = false, &block)
    nrows.times do |row|
      yield self[row]
    end
  end

  # Iterates through each Flask view of a row in a Jug.
  # Iterates in increasing order, and returns views of
  # each Flask, as well as the corresponding row index
  def each_row_index(*, all = false, &block)
    nrows.times do |row|
      yield self[row], row
    end
  end

  # Iterates through each Flask view of a column in a Jug.
  # Iterates in increasing order, and returns views of
  # each Flask.
  def each_col(*, all = false, &block)
    ncols.times do |col|
      yield self[..., col]
    end
  end

  # Iterates through each Flask view of a column in a Jug.
  # Iterates in increasing order, and returns views of
  # each Flask, as well as the corresponding column index
  def each_col_index(*, all = false, &block)
    ncols.times do |col|
      yield self[..., col], col
    end
  end

  # Allocates a slice for `nrows` and `ncols`, and returns
  # an empty Jug with no values set.  Dtype must be specified
  # when creating an empty Jug
  #
  # ```
  # j = Jug(Int32).empty(3, 3)
  # j # => [[0, 0, 0]
  #         [0, 0, 0]
  #         [0, 0, 0]]
  # ```
  def self.empty(nrows, ncols, dtype : U.class = Float64) forall U
    Jug(U).new Slice(U).new(nrows * ncols), nrows, ncols, ncols
  end

  def self.random(r : Range(U, U), nrows, ncols) forall U
    Jug(U).new(nrows, ncols) do |_, _|
      Random.rand(r)
    end
  end

  # Returns a flattened Flask view of a Jug.
  #
  # ```
  # j = Jug.new [[1, 2, 3], [4, 5, 6]]
  # j.ravel # => [1, 2, 3, 4, 5, 6]
  # ```
  def ravel
    Flask(T).new data, nrows * ncols, 1
  end

  # Pours a Flask from a Jug row
  #
  # ```
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[0] # => [1, 2, 3]
  # ```
  def [](i : Indexer)
    slice = data[*stride_offset_row(i, tda, ncols, 1)]
    Flask.new slice, ncols, 1
  end

  # Pours a Flask into a Jug row
  #
  # ```
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[2] = [1, 2, 3]
  # ```
  def []=(i : Indexer, row : Flask(T))
    ncols.times do |j|
      self[i, j] = row[j]
    end
  end

  # Pours a Flask into a Jug column
  #
  # ```
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[..., 2] = [1, 2, 3]
  # ```
  def []=(i : Range(Nil, Nil), j : Indexer, col : Flask(T))
    nrows.times do |r|
      self[r, j] = col[r]
    end
  end

  # Pours a drop from a Jug
  #
  # ```
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[2, 2] # => 9
  # ```
  def [](i : Indexer, j : Indexer)
    data[stride_offset(i, tda, j, 1)]
  end

  # Pours a drop into a Jug
  #
  # ```
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[2, 2] = 100
  # ```
  def []=(i : Indexer, j : Indexer, x : T)
    data[stride_offset(i, tda, j, 1)] = x
  end

  # Pours a Flask from a Jug column
  #
  # ```
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[..., 0] # => [1, 4, 7]
  # ```
  def [](i : Range(Indexer?, Indexer?), j : Indexer)
    x = convert_range_to_slice(
      i, nrows)
    x_offset = x.end - x.begin
    xi = x.end - 1
    start = stride_offset(
      x.begin, tda, j, 1)
    finish = stride_offset(
      xi, tda, ncols, 1)
    slice = data[start, finish - start]

    Flask(T).new(
      slice,
      x_offset.to_i32,
      tda,
    )
  end

  # Pours a view of a jug from given ranges
  #
  # ```
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m[...2, ...2] # => [[1, 2], [4, 5]]
  # ```
  def [](i : Range(Indexer?, Indexer?) = ..., j : Range(Indexer?, Indexer?) = ...)
    x = convert_range_to_slice(
      i, nrows)
    y = convert_range_to_slice(
      j, ncols)
    x_offset = x.end - x.begin
    y_offset = y.end - y.begin
    xi = x.end - 1
    yi = y.end - 1
    start = stride_offset(
      x.begin, tda, y.begin, 1)
    finish = stride_offset(
      xi, tda, yi, 1) + 1
    slice = data[start, finish - start]
    rows = x_offset.to_i32
    cols = y_offset.to_i32

    Jug(T).new(
      slice,
      rows,
      cols,
      tda,
    )
  end

  def matmul(other : Jug(T))
    LL.matmul(self.clone, other)
  end

  def inv
    LA.inv(self.clone)
  end

  def tril(k = 0)
    Jug(T).new(nrows, ncols) do |i, j|
      i < j - k ? self[i, j] : T.new(0)
    end
  end

  def triu(k = 0)
    Jug(T).new(nrows, ncols) do |i, j|
      i > j - k ? self[i, j] : T.new(0)
    end
  end

  def self.identity(n : Int32)
    one = LL.astype(1, T)
    zero = LL.astype(0, T)
    Jug(T).new(n, n) do |i, j|
      i == j ? T.new(1) : T.new(0)
    end
  end

  def diagonal
    n = Math.min(nrows, ncols)
    Flask(T).new(n) { |i| self[i, i] }
  end

  def trace
    diagonal.sum
  end

  # Elementwise addition of a Flask to another equally sized Flask
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 + j1 # => [[2, 4], [6, 8]]
  # ```
  def +(other : Jug(T))
    LL.add(self.clone, other)
  end

  # Elementwise addition of a Flask to a scalar
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 + 2 # => [[3, 4], [5, 6]]
  # ```
  def +(other : T)
    LL.add(self.clone, other)
  end

  # Elementwise subtraction of a Flask to another equally sized Flask
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 - j1 # => [[0, 0], [0, 0]]
  # ```
  def -(other : Jug(T))
    LL.sub(self.clone, other)
  end

  # Elementwise subtraction of a Flask with a scalar
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 - 2 # => [[-1, 0], [1, 2]]
  # ```
  def -(other : T)
    LL.sub(self.clone, other)
  end

  # Elementwise multiplication of a Flask to another equally sized Flask
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 * j1 # => [[1, 4], [9, 16]]
  # ```
  def *(other : Jug(T))
    LL.mul(self.clone, other)
  end

  # Elementwise multiplication of a Flask to a scalar
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 * 2 # => [[2, 4], [6, 8]]
  # ```
  def *(other : T)
    LL.mul(self.clone, other)
  end

  # Elementwise division of a Flask to another equally sized Flask
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 / j1 # => [[1, 1], [1, 1]]
  # ```
  def /(other : Jug(T))
    LL.div(self.clone, other)
  end

  # Elementwise division of a Flask to a scalar
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 / 2 # => [[0.5, 1.0], [1.5, 2.0]]
  # ```
  def /(other : T)
    LL.div(self.clone, other)
  end

  def >(other : Jug)
    MA.gt(self, other)
  end

  def >(other : Number)
    MA.gt(self, other)
  end

  def >=(other : Jug)
    MA.ge(self, other)
  end

  def >=(other : Number)
    MA.ge(self, other)
  end

  def <(other : Jug)
    MA.lt(self, other)
  end

  def <(other : Number)
    MA.lt(self, other)
  end

  def <=(other : Jug)
    MA.le(self, other)
  end

  def <=(other : Number)
    MA.le(self, other)
  end

  def ==(other : Jug)
    MA.eq(self, other)
  end

  def ==(other : Number)
    MA.eq(self, other)
  end

  # Accumulates an operation along a row or column of a Jug.
  # This returns an instance of an Accumulate2D which provides
  # the definitions for many cumulative operations axis-wise
  # along a Jug.  This is one of the many ufuncs that Bottle provides
  #
  # ```crystal
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m.accumulate(axis = 1).add
  # m # => [[1, 3, 6], [4, 9, 15], [7, 15, 24]]
  # ```
  def accumulate(axis : Int32, inplace = false)
    Accumulate2D.new(self, axis, inplace)
  end

  # Computes the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max
    ravel.max
  end

  def max(axis : Indexer)
    reduce(axis, &.max)
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    ravel.min
  end

  def min(axis : Indexer)
    reduce(axis, &.max)
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    {min, max}
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    max - min
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmax # => 3
  # ```
  def argmax
    ravel.argmax
  end

  def argmax(axis : Indexer)
    reduce(axis, &.argmax)
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmin # => 0
  # ```
  def argmin
    ravel.argmin
  end

  def argmin(axis : Indexer)
    reduce(axis, &.argmin)
  end
end
