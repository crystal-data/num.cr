require "random"
require "./arithmetic"
require "./indexing"
require "../libs/dtype"
require "../ma/mask"

class Flask(T)
  include Bottle::Internal::Dtype
  include Bottle::Internal::Indexing
  include Bottle

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
    io << "[" << (0...size).map { |i| self[i].round(3) }.join(", ") << "]"
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
  # of data.  This is just another alias for zeros,
  # since Crystal right now doesn't support slices
  # pointing to generic memory, but if Crystal does
  # support this down the road, this will change
  #
  # ```crystal
  # f = Flask.empty(5, dtype = Int32)
  # f # => [0, 0, 0, 0, 0]
  # ```
  def self.empty(n : Indexer, dtype : U.class = Float64) forall U
    Flask(U).new Slice(U).new(n), n, 1
  end

  # Pours a flask full of zeros.  Default
  # dtype is Float64, but any dtype that
  # can holds zeros is supported.
  #
  # ```crystal
  # f = Flask.zeros(5, dtype = Int32)
  # f # => [0, 0, 0, 0, 0]
  # ```
  def self.zeros(n : Indexer, dtype : U.class = Float64) forall U
    Flask(U).new Slice(U).new(n), n, 1
  end

  # Pours a flask full of ones.  Default
  # dtype is Float64, but any dtype that
  # can holds zeros is supported.
  #
  # ```crystal
  # f = Flask.ones(5, dtype = Int32)
  # f # => [1, 1, 1, 1, 1]
  # ```
  def self.ones(n : Indexer, dtype : U.class = Float64) forall U
    Flask(U).new Slice(U).new(n, U.new(1)), n, 1
  end

  # Pours a flask full of a given scalar.  Default
  # dtype is Float64, but any dtype that
  # can holds numbers is supported.
  #
  # ```crystal
  # f = Flask.full(5, 5, dtype = Int32)
  # f # => [5, 5, 5, 5, 5]
  # ```
  def self.full(n : Indexer, x : Number, dtype : U.class = Float64) forall U
    Flask(U).new Slice(U).new(n, U.new(x)), n, 1
  end

  # Pours a flask full of random data.
  # The dtype of the flask is inferred
  # from the values on either end of the
  # range.
  #
  # ```crystal
  # f = Flask.random(0, 10, 5)
  # f # => [4, 8, 7, 8, 4]
  # ```
  def self.random(r : Range(U, U), n : Int32) forall U
    Flask(U).new(n) { |_| Random.rand(r) }
  end

  # Pours a flask into a flask of
  # a different dtype.  The cast will
  # be successful if the types `new` method
  # takes the old value
  #
  # ```crystal
  # f = Flask.new [1, 2, 3]
  # f.astype(Float64) # => [1.0, 2.0, 3.0]
  # ```
  def astype(dtype : U.class) forall U
    if T == U
      return self
    end
    Flask(U).new(self.size) { |i| U.new(self[i]) }
  end

  # Gets a single element from a Flask at a given index, the core
  # indexing operation of a Flask
  #
  # ```
  # f = Flask.new [1, 2, 3, 4, 5]
  # f[0] # => 1
  # ```
  def [](index : Indexer)
    data[stride_offset(index, stride)]
  end

  # Gets multiple elements from a Flask at given indexes.  This returns
  # a `copy` since there is no way to create a contiguous slice of memory
  #
  # ```
  # f = Flask.new [1, 2, 3]
  # f[[1, 2]] # => [2, 3]
  # ```
  def [](indexes : Array(Indexer))
    Flask.new indexes.map { |i| self[i] }
  end

  # Returns a view of a Flask defined by a given range.  Currently only
  # supports single strided ranges due to limitations of Crystal
  #
  # ```
  # f = Flask.new [1, 2, 3, 4, 5]
  # f[2...4] # => [3, 4]
  # ```
  def [](range : Range(Indexer?, Indexer?))
    rng = LL.convert_range_to_slice(range, size)
    Flask.new data[rng.begin, rng.end - rng.begin], size, stride
  end

  # Sets a single element from a Flask at a given index
  #
  # ```
  # f = Flask.new [1, 2, 3]
  # f[0] = 10
  # f # => [10, 2, 3]
  # ```
  def []=(index : Indexer, value : Number)
    data[stride_offset(index, stride)] = value
  end

  # Sets multiple elements of a Flask by the given indexes.
  #
  # ```
  # f = Flask.new [1, 2, 3]
  # f[[0, 1]] = [10, 9]
  # f # => [10, 9, 3]
  # ```
  def []=(indexes : Array(Indexer), values : Array(Number))
    indexes.each_with_index { |e, i| self[e] = values[i] }
  end

  # Sets elements of a Flask to given values based on the given range
  #
  # ```
  # f = Flask.new [1, 2, 3, 4, 5]
  # f[1...] = [10, 9, 8, 7]
  # f # => [1, 10, 9, 8, 7]
  # ```
  def []=(range : Range(Indexer?, Indexer?), values : Array(Number))
    range = LL.convert_range_to_slice(range, size)
    range.each_with_index { |e, i| self[e] = values[i] }
  end

  # Computes the dot product of two vectors
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [4.0, 5.0, 6.0]
  # f1.dot(f2) # => 32
  # ```
  def dot(other : Flask(T))
    LL.dot(self, other)
  end

  # Computes the euclidean norm of the vector
  #
  # ```
  # vec = Flask.new [1.0, 2.0, 3.0]
  # vec.norm # => 3.741657386773941
  # ```
  def norm
    LL.norm(self)
  end

  # Sum of absolute values
  #
  # ```
  # f1 = Flask.new [-1, 1, 2]
  # f2.asum # => 4
  # ```
  def asum
    LL.asum(self)
  end

  # Index of absolute value max
  #
  # ```
  # f1 = Flask.new [-8, 1, 2]
  # f2.amax # => 0
  # ```
  def amax
    LL.amax(self)
  end

  # Elementwise addition of a Flask to another equally sized Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [2.0, 4.0, 6.0]
  # f1 + f2 # => [3.0, 6.0, 9.0]
  # ```
  def +(other : Flask(T))
    LL.add(self.clone, other)
  end

  # Elementwise addition of a Flask to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [3.0, 4.0, 5.0]
  # ```
  def +(other : T)
    LL.add(self.clone, other)
  end

  # Elementwise subtraction of a Flask to another equally sized Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [2.0, 4.0, 6.0]
  # f1 - f2 # => [-1.0, -2.0, -3.0]
  # ```
  def -(other : Flask(T))
    LL.sub(self.clone, other)
  end

  # Elementwise subtraction of a Flask with a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 - f2 # => [-1.0, 0.0, 1.0]
  # ```
  def -(other : T)
    LL.sub(self.clone, other)
  end

  # Elementwise multiplication of a Flask to another equally sized Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [2.0, 4.0, 6.0]
  # f1 * f2 # => [3.0, 8.0, 18.0]
  # ```
  def *(other : Flask(T))
    LL.mul(self.clone, other)
  end

  # Elementwise multiplication of a Flask to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [2.0, 4.0, 6.0]
  # ```
  def *(other : T)
    LL.mul(self.clone, other)
  end

  # Elementwise division of a Flask to another equally sized Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [2.0, 4.0, 6.0]
  # f1 / f2 # => [0.5, 0.5, 0.5]
  # ```
  def /(other : Flask(T))
    LL.div(self.clone, other)
  end

  # Elementwise division of a Flask to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 / f2 # => [0.5, 1, 1.5]
  # ```
  def /(other : T)
    LL.div(self.clone, other)
  end

  # Elementwise greater than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 > f2 # => [false, false, true ]
  # ```
  def >(other : Flask)
    LL.gt(self, other)
  end

  # Elementwise greater than comparison to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 > f2 # => [false, false, true ]
  # ```
  def >(other : Number)
    LL.gt(self, other)
  end

  # Elementwise greater equal than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 >= f2 # => [false, true, true ]
  # ```
  def >=(other : Flask)
    LL.ge(self, other)
  end

  # Elementwise greater equal than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 >= f2 # => [false, true, true ]
  # ```
  def >=(other : Number)
    LL.ge(self, other)
  end

  # Elementwise less than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 < f2 # => [true, false, false]
  # ```
  def <(other : Flask)
    LL.lt(self, other)
  end

  # Elementwise less than comparison to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 < f2 # => [false, false, true ]
  # ```
  def <(other : Number)
    LL.lt(self, other)
  end

  # Elementwise less equal than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 <= f2 # => [true, true, false]
  # ```
  def <=(other : Flask)
    LL.le(self, other)
  end

  # Elementwise less equal than comparison to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 > f2 # => [true, true, false]
  # ```
  def <=(other : Number)
    LL.le(self, other)
  end

  # Elementwise equal comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 == f2 # => [false, true, false ]
  # ```
  def ==(other : Flask)
    LL.eq(self, other)
  end

  # Elementwise equal comparison to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 > f2 # => [false, true, false]
  # ```
  def ==(other : Number)
    LL.eq(self, other)
  end

  # Sum reduction for a Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f1.sum # => 6.0
  # ```
  def sum
    reduce { |i, j| i + j }
  end

  # Product reduction for a Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 4.0]
  # f1.prod # => 8.0
  # ```
  def prod
    reduce { |i, j| i * j }
  end

  # Computes the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max
    _, max, _ = max_internal
    max
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmax # => 3
  # ```
  def argmax
    _, _, index = max_internal
    index
  end

  # Internal method to find the maximum value and the index
  # of the maximum value for a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max_internal # => {true, 4, 3}
  # ```
  private def max_internal
    max = uninitialized T
    index = uninitialized Int32
    found = false

    each_with_index do |elem, i|
      if i == 0 || elem > max
        max = elem
        index = i
      end
      found = true
    end

    {found, max, index}
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    _, min, _ = min_internal
    min
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmin # => 0
  # ```
  def argmin
    _, _, index = min_internal
    index
  end

  # Internal method to find the maximum value and the index
  # of the maximum value for a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max_internal # => {true, 4, 3}
  # ```
  private def min_internal
    min = uninitialized T
    index = uninitialized Int32
    found = false

    each_with_index do |elem, i|
      if i == 0 || elem < min
        min = elem
        index = i
      end
      found = true
    end

    {found, min, index}
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    _, min, max, _, _ = ptp_internal
    return {min, max}
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    min, max = ptpv
    return max - min
  end

  # Internal method to find the minimum and maximum values,
  # as well as the respective indexes for a flask.
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp_internal # => {true, 1, 4, 0, 3}
  # ```
  private def ptp_internal
    min = uninitialized T
    max = uninitialized T
    imin = uninitialized Int32
    imax = uninitialized Int32
    found = false

    each_with_index do |elem, i|
      if i == 0 || elem < min
        min = elem
        imin = i
      end
      if i == 0 || elem > max
        max = elem
        imax = i
      end
      found = true
    end
    {found, min, max, imin, imax}
  end

  # Computes the cumulative sum of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumsum # => [1, 3, 6, 10]
  # ```
  def cumsum
    ret = self.clone
    ret.cumsum!
    ret
  end

  # Computes the cumulative sum of a vector in place.
  # Primarily used for reductions along an axis in
  # a Jug.
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumsum!
  # v # => [1, 3, 6, 10]
  # ```
  def cumsum!
    (1...size).each do |i|
      self[i] += self[i - 1]
    end
  end

  # Computes the cumulative product of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumprod # => [1, 2, 6, 24]
  # ```
  def cumprod
    ret = self.clone
    ret.cumprod!
    ret
  end

  # Computes the cumulative product of a vector in place.
  # Primarily used for reductions along an axis in
  # a Jug.
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumprod!
  # v # => [1, 2, 6, 24]
  # ```
  def cumprod!
    (1...size).each do |i|
      self[i] *= self[i - 1]
    end
  end
end
