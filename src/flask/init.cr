require "../libs/dtype"
require "./*"
require "../blas/*"
require "../jug/*"

class Flask(T)
  getter data : Slice(T)
  getter size : Int32
  getter stride : Int32

  # Converts a flask into a string representation.  This currently
  # displays all the values of a flask, but should re-worked to
  # truncate large flasks
  #
  # ```
  # f = Flask.new [1, 2, 3, 4, 5]
  # puts f # => flask[1, 2, 3, 4, 5]
  # ```
  def to_s(io)
    io << "[" << (0...size).map { |i| self[i] }.join(", ") << "]"
  end

  # Returns a copy of a flask that owns its own memory
  #
  # ```
  # f = Flask.new [1, 2, 3, 4, 5]
  # f.clone # => [1, 2, 3, 4, 5]
  # ```
  def clone
    Flask.new data.dup, size, stride
  end

  # Reverses a copy of the flask and returns the copy.
  # TODO: Implement negative strides for an array
  # ```
  # f = Flask.new [1, 2, 3, 4, 5]
  # f.reverse # => [5, 4, 3, 2, 1]
  # ```
  def reverse
    f = clone
    f.data.reverse!
    return f
  end

  def reshape(rows : Indexer, cols : Indexer)
    Jug(T).new(
      data,
      rows,
      cols,
      cols * stride,
      stride
    )
  end

  # Initializes a flask from an Indexable of a type.
  # this is the common user facing init function, and
  # allocates a slice and sets its data to the elements
  # of data
  def initialize(data : Indexable(T))
    @size = data.size
    @stride = 1
    @data = Slice(T).new(size) { |i| data[i] }
  end

  # Primarily a convenience method to allow for cloning
  # of Vectors, should not be called by outside methods.
  def initialize(@data : Slice(T), @size, @stride)
  end

  # Yields each index of a flask, useful for providing a base
  # method to other iterators through a flask
  def each_index(*, all = false, &block)
    size.times do |i|
      yield i
    end
  end

  # Lazily yields each element of a flask
  def each(*, all = false, &block)
    each_index(all: all) { |i| yield self[i] }
  end

  # Lazily yields each element of a flask, along with its
  # index.
  def each_with_index(*, all = false, &block)
    each_index(all: all) { |i| yield(self[i], i) }
  end
end

f = Flask.new [1, 2, 3, 4, 5, 6]
puts f.reshape(3, 2)
