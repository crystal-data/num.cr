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
end
