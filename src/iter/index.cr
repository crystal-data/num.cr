require "../base/base"

struct Num::Iter::IndexIter
  include Iterator(Array(Int32))
  @ndims : Int32
  @track : Array(Int32)
  @shape : Array(Int32)

  def initialize(arr : BaseArray(T))
    @shape = arr.shape.dup
    @track = [0] * @shape.size
    @ndims = @shape.size - 1
  end

  def next
    if @done
      return stop
    end
    last = @track.dup

    @ndims.step(to: 0, by: -1) do |i|
      @track[i] += 1
      if @track[i] == @shape[i]
        if i == 0
          @done = true
        end
        @track[i] = 0
        next
      end
      break
    end
    last
  end
end
