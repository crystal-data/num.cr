require "../core/ndtensor"

struct Bottle::Internal::ContigIter(T)
  include Iterator(T)
  @t : Tensor(T)
  @idx : Array(Int32)
  @done : Bool = false
  @p_item : Pointer(T)
  @transpose : Bool

  def initialize(@t : Tensor(T), @transpose : Bool = false)
    @idx = [0] * @t.ndims
    @p_item = @t.@buffer
  end

  def next
    if @done
      return stop
    end
    last = @p_item
    @transpose ? _increment_idx_trans : _increment_idx
    last
  end

  private def _increment_idx_trans
    if (@done)
      return false
    end
    ii = 0
    @idx[ii] += 1
    while (@idx[ii] == @t.shape[ii])
      # reset ith to zero  019 -> 010
      @idx[ii] = 0
      # bump up next most significant by one  010 --> 020
      ii += 1

      if (ii == @t.ndims)
        @done = true
        return false
      end
      @idx[ii] += 1
    end
    @p_item = @t.ptr_at(@idx)
    true
  end

  private def _increment_idx
    if (@done)
      return false
    end
    ii = @idx.size - 1
    @idx[ii] += 1

    while (@idx[ii] == @t.shape[ii])
      @idx[ii] = 0
      ii -= 1
      if (ii < 0)
        @done = true
        return false
      end
      @idx[ii] += 1
    end
    @p_item = @t.ptr_at(@idx)
    true
  end
end
