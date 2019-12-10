require "../base/base"

struct Num::Internal::PermuteIter(T)
  include Iterator(T)

  @shape : Array(Int32)
  @ranges : Array(Int32 | Range(Int32, Int32))
  @ret : Array(Int32 | Range(Int32, Int32))
  @done : Bool = false
  @axis : Int32
  @dims : Int32

  def initialize(t : BaseArray(T), @axis : Int32)
    @shape = t.shape.dup
    @ranges = @shape.map_with_index do |_, i|
      i == @axis ? 0...@shape[@axis] : 0
    end
    @ret = @ranges.dup
    @dims = t.ndims - 1
    @min_axis = @axis == 0 ? 1 : 0
  end

  def next
    if @done
      return stop
    end
    @ret = @ranges.dup
    @dims.step(to: 0, by: -1) do |i|
      if i == @axis
        next
      end

      ri = @ranges[i].as(Int32)
      @ranges[i] = ri + 1

      if @ranges[i] == @shape[i]
        if i == @min_axis
          @done = true
        end
        @ranges[i] = 0
        next
      end
      break
    end
    @ret
  end
end
