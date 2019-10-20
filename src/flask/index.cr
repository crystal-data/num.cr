require "./*"
require "../indexing/base"
require "../libs/dtype"
require "../strides/offsets"

class Flask(T)
  # Gets a single element from a Flask at a given index, the core
  # indexing operation of a Flask
  #
  # ```
  # f = Flask.new [1, 2, 3, 4, 5]
  # f[0] # => 1
  # ```
  def [](index : Indexer)
    data[Strides.offset(index, stride)]
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
    data[Strides.offset(index, stride)] = value
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
end
