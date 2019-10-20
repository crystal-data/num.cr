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

  def each_index(*, all = false, &block)
    nrows.times do |i|
      ncols.times do |j|
        yield i, j
      end
    end
  end

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

  def each(*, all = false, &block)
    each_index(all: all) { |i, j| yield self[i, j] }
  end

  def each_with_index(*, all = false, &block)
    each_index(all: all) { |i, j| yield(self[i, j], i, j) }
  end

  def each_row(*, all = false, &block)
    nrows.times do |row|
      yield self[row]
    end
  end

  def each_row_index(*, all = false, &block)
    nrows.times do |row|
      yield self[row], row
    end
  end

  def each_col(*, all = false, &block)
    ncols.times do |col|
      yield self[..., col]
    end
  end

  def each_col_index(*, all = false, &block)
    ncols.times do |col|
      yield self[..., col], col
    end
  end

  def self.empty(nrows, ncols)
    Jug(T).new Slice(T).new(nrows * ncols), nrows, ncols, ncols, 1
  end

  def ravel
    Flask(T).new data, nrows * ncols, jstride
  end
end
