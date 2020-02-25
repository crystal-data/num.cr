require "../tensor/tensor"

module Num
  def average(a)
    mean(a)
  end

  def average(a, axis : Int32)
    mean(a, axis: axis)
  end

  def extended_euclidean(a : Int32, b : Int32)
    s = 0
    old_s = 1
    t = 1
    old_t = 0
    r = b
    old_r = a

    while r != 0
      quotient = old_r // r
      old_r, r = r, old_r - quotient * r
      ols_s, s = s, old_s - quotient * s
      old_t, t = t, old_t - quotient * t
    end

    {old_s, old_t, old_r, t, s}
  end

  def gcde(a : Tensor(Int32), b : Tensor(Int32))
    a, b = NumInternal.broadcast2(a, b)
    ret_shape = a.shape + [5]
    ret = Tensor(Int32).new(ret_shape)
    ret_iter = ret.unsafe_iter
    a.iter2(b).each do |i, j|
      tmp = extended_euclidean(i.value, j.value)
      5.times do |n|
        ret_iter.next.value = tmp[n]
      end
    end
    ret
  end
end
