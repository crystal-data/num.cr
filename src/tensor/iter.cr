require "./tensor"

struct NumInternal::MatrixIter(T)
  include Iterator(T)
  getter tns : Tensor(T)
  getter axis : Int32
  getter dim : Int32
  getter ranges : Array(Int32 | Range(Int32, Int32))
  @idx : Int32 = 0

  def initialize(@tns : Tensor(T))
    if tns.ndims < 3
      raise "Dimensionality of the Tensor is not high enough to reduce"
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
    tns.slice(@ranges)
  end
end
