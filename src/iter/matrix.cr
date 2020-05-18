require "../array/array"

struct NumInternal::MatrixIter(T)
  include Iterator(T)
  getter tns : AnyArray(T)
  getter axis : Int32
  getter dim : Int32
  getter ranges : Array(Int32 | Range(Int32, Int32))
  @idx : Int32 = 0

  def initialize(@tns : AnyArray(T))
    if tns.ndims < 3
      raise "Dimensionality of the Array is not high enough to reduce"
    end

    @axis = tns.ndims - 3
    @dim = tns.shape[axis]
    @ranges = tns.shape.map_with_index do |a, i|
      axis == i ? 0 : 0...a
    end
  end

  def next
    if @idx == @dim
      return stop
    end
    @ranges[axis] = @idx
    @idx += 1
    tns[@ranges]
  end
end
