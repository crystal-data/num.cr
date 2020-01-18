require "./macros"
require "./common"

module Num
  def where(condition : BaseArray(Bool), x : BaseArray(U), y : BaseArray(U)) forall U
    broadcast_rhs condition, x
    broadcast_rhs condition, y
    ret = Tensor(U).new(condition.shape)
    ret.flat_iter.zip(condition.flat_iter, x.flat_iter, y.flat_iter) do |v, i, j, k|
      v.value = i.value ? j.value : k.value
    end
    ret
  end

  def where(condition : BaseArray(Bool), x : BaseArray)
    broadcast_rhs condition, x
    ret = Tensor(Float64).new(condition.shape)
    ret.flat_iter.zip(condition.flat_iter, x.flat_iter) do |v, i, j|
      v.value = i.value ? Float64.new(j.value) : Float64::NAN
    end
    ret
  end

  def where(condition : BaseArray(Bool))
    ret = Pointer(Int32).malloc(condition.size * condition.ndims)
    ptr = ret
    offset = 0
    n = condition.ndims
    condition.flat_iter.zip(condition.index_iter) do |b, i|
      if b.value
        i.to_unsafe.move_to(ptr, n)
        offset += n
        ptr += n
      end
    end
    Tensor(Int32).new(ret, [offset // n, n], [n, 1], ArrayFlags::Contiguous, nil)
  end

  def nonzero(condition : BaseArray(U)) forall U
    if U != Bool
      condition = condition == 0
    end
    where(condition)
  end

  def flatnonzero(array : BaseArray(U)) forall U
    if U != Bool
      array = array == 0
    end
    offset = 0
    idx = 0

    ptr = Pointer(Int32).malloc(array.size)

    array.flat_iter.each do |i|
      if i.value
        ptr[offset] = idx
        offset += 1
      end
      idx += 1
    end

    ptr = ptr.realloc(offset)
    array.basetype.new([offset]) { |i| ptr[i] }
  end
end
