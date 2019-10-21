require "./*"
require "../libs/dtype"
require "../flask/*"
require "../strides/offsets"
require "../indexing/base"

class Jug(T)
  getter data : Slice(T)
  getter nrows : Int32
  getter ncols : Int32
  getter istride : Int32
  getter jstride : Int32

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
    Jug(T).new data.dup, nrows, ncols, istride, jstride
  end

  # Initializes a Jug from an Indexable of a type.
  # this is the common user facing init function, and
  # allocates a slice and sets its data to the elements
  # of data
  def initialize(data : Indexable(Indexable(T)))
    @nrows = data.size
    @ncols = data[0].size
    @istride = ncols
    @jstride = 1
    @data = Slice(T).new(nrows * ncols) do |idx|
      i = idx // ncols
      j = idx % ncols
      data[i][j]
    end
  end

  # Primarily a convenience method to allow for cloning
  # of Vectors, should not be called by outside methods.
  def initialize(@data, @nrows, @ncols, @istride, @jstride)
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
    ary = Flask(U).empty(axis == 0 ? ncols : nrows)
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

  # Accumulates an operation along a row or column of a Jug.
  # Used to provide cumulative operations such as cumsum or
  # cumprod.  The most common accumulations have aliases on the
  # class.  For example, `m.cumsum(0)` is equivalent to
  # m.accumulate(0, &.cumsum!).  Accumulations are always
  # done IN PLACE on the Jug, and all aliases on the class
  # clone the jug first.
  #
  # ```crystal
  # m = Jug.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m.accumulate(1, &.cumsum!)
  # m # => [[1, 3, 6], [4, 9, 15], [7, 15, 24]]
  # ```
  def accumulate(axis : Indexer, &block : Flask(T) -> Nil)
    if axis == 0
      each_col_index do |e, c|
        yield e
      end
    else
      each_row_index do |e, r|
        yield e
      end
    end
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
  def self.empty(nrows, ncols)
    Jug(T).new Slice(T).new(nrows * ncols), nrows, ncols, ncols, 1
  end

  # Returns a flattened Flask view of a Jug.
  #
  # ```
  # j = Jug.new [[1, 2, 3], [4, 5, 6]]
  # j.ravel # => [1, 2, 3, 4, 5, 6]
  # ```
  def ravel
    Flask(T).new data, nrows * ncols, jstride
  end
end
