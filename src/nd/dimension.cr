# Represents the shape of an Array.  This is an
# immutable attribute since NDArrays have a fixed
# size.  Contains the capability to identify
# the strides of an array from the shape.
#
# ```
# dim = Shape.new([2, 4, 2])
# dim._2s # => Strides(@dims=[8, 2, 1])
# dim.totalsize # => 16
# ```
struct Shape
  getter dims : Array(Int32)

  # Takes an array of Int32 and stores
  # it as the shape of an NDArray
  #
  # ```
  # Shape.new([1, 4, 5])
  # ```
  def initialize(@dims : Array(Int32))
  end

  # Passes missing methods to the shape array
  # to facilitate using Array methods directly
  # on the `Shape` struct.
  forward_missing_to @dims

  # :nodoc:
  def strides
    cumsum = 1
    Strides.new(([1] + self.reverse[...-1]).map do |i|
      ret = i * cumsum
      cumsum *= i
    end.reverse)
  end

  # Returns the total number of elements in an
  # NDArray.
  #
  # ```
  # dim = Shape.new([2, 4, 2])
  # dim.totalsize # => 16
  # ```
  def totalsize
    self.reduce { |i, j| i * j }
  end
end

# Tracks of the number of elements between
# each dimension of an NDArray, useful for
# traversing the contiguous memory block
# associated with an NDArray
struct Strides
  getter dims : Array(Int32)

  # Constructs a strides object from
  # an Array of Integers
  def initialize(@dims)
  end

  forward_missing_to @dims
end
