require "../libs/dtype"
require "./*"
require "../blas/*"
require "../jug/*"
require "random"

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

  # Initializes a flask from an Indexable of a type.
  # this is the common user facing init function, and
  # allocates a slice and sets its data to the elements
  # of data
  def initialize(data : Indexable(T))
    @size = data.size
    @stride = 1
    @data = Slice(T).new(size) { |i| data[i] }
  end

  def self.new(size : Int32, &block)
    data = Slice(T).new(size) { |i| yield i }
    new(data, size, 1)
  end

  # Primarily a convenience method to allow for cloning
  # of Vectors, should not be called by outside methods.
  def initialize(@data : Slice(T), @size, @stride)
  end

  # Applies a reduction operation to a flask to convert
  # a flask into a scalar. Many common reductions are
  # aliased on the class, such as sum and prod.  This
  # is for convenience only, the performance is the
  # same as writing the reduction.
  #
  # ```crystal
  # f = Flask.new [1, 2, 3]
  # f.reduce { |i, j| i + j } # => 6
  # ```
  def reduce(&block : T -> T)
    memo = uninitialized T
    found = false

    each do |elem|
      memo = found ? (yield memo, elem) : elem
      found = true
    end

    found ? memo : raise Enumerable::EmptyError.new
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

  # Checks if `false` is returned by any values of the
  # passed block, otherwise returns true.
  #
  # ```crystal
  # f = Flask.new [1, 5, 9]
  # f.all { |i| i > 1 } # => false
  # ```
  def all?
    each { |e| return false unless yield e }
    true
  end

  # checks if `false` is explicitly in the flask
  #
  # ```crystal
  # f = Flask.new [true, true, true]
  # f.all? # => true
  # ```
  def all?
    all? &.itself
  end

  # Initializes a Flask with an uninitialized slice
  # of data
  def self.empty(n : Indexer)
    Flask(T).new Slice(T).new(n), n, 1
  end

  # Initializes a flask containing random data using
  # the provided ranges and size.  The dtype of the ranges
  # determines the output type of the flask.
  def self.random(r : Range(U, U), n : Int32) forall U
    Flask(U).new(n) { |_| Random.rand(r) }
  end

  # Converts a Flask to a different dtype if the cast
  # can be made.
  def astype(dtype : U.class) forall U
    if T == U
      return self
    end
    Flask(U).new(self.size) { |i| U.new(self[i]) }
  end
end
