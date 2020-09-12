require "../tensor"

macro init_strided_iteration(coord, backstrides, t_shape, t_strides, t_rank)
  {{ coord.id }} = Pointer(Int32).malloc({{ t_rank }}, 0)
  {{ backstrides.id }} = Pointer(Int32).malloc({{ t_rank }})
  {{ t_rank }}.times do |i|
    {{ backstrides.id }}[i] = {{ t_strides }}[i] * ({{ t_shape }}[i] - 1)
  end
end

macro advance_strided_iteration(coord, backstrides, t_shape, t_strides, t_rank, iter_pos)
  ({{ t_rank }} - 1).step(to: 0, by: -1) do |k|
    if {{ coord.id }}[k] < {{ t_shape }}[k] - 1
      {{ coord.id }}[k] += 1
      {{ iter_pos }} += {{ t_strides }}[k]
      break
    else
      {{ coord.id }}[k] = 0
      {{ iter_pos }} -= {{ backstrides.id }}[k]
    end
  end
end

@[AlwaysInline]
def strided_iteration(t : Tensor)
  data = t.to_unsafe
  if t.flags.contiguous?
    t.size.times do
      yield data
      data += 1
    end
  else
    t_shape, t_strides, t_rank = t.iter_attrs
    init_strided_iteration(:coord, :backstrides, t_shape, t_strides, t_rank)
    t.size.times do
      yield data
      advance_strided_iteration(:coord, :backstrides, t_shape, t_strides, t_rank, data)
    end
  end
end

@[AlwaysInline]
def dual_strided_iteration(t1 : Tensor, t2 : Tensor)
  t1_contiguous = t1.flags.contiguous?
  t2_contiguous = t2.flags.contiguous?

  t1data = t1.to_unsafe
  t2data = t2.to_unsafe

  if t1_contiguous && t2_contiguous
    t1.size.times do
      yield t1data, t2data
      t1data += 1
      t2data += 1
    end
  elsif t1_contiguous
    t2_shape, t2_strides, t2_rank = t2.iter_attrs
    init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank)
    t1.size.times do
      yield t1data, t2data
      t1data += 1
      advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
    end
  elsif t2_contiguous
    t1_shape, t1_strides, t1_rank = t1.iter_attrs
    init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank)
    t1.size.times do
      yield t1data, t2data
      advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      t2data += 1
    end
  else
    t1_shape, t1_strides, t1_rank = t1.iter_attrs
    t2_shape, t2_strides, t2_rank = t2.iter_attrs
    init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank)
    init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank)
    t1.size.times do
      yield t1data, t2data
      advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
    end
  end
end

def map(a : Tensor(U), &block : U -> V) forall U, V
  ret = Tensor(V).new([a.size])
  dual_strided_iteration(a, ret) do |i, j|
    j.value = yield i.value
  end
  ret
end

def map!(a : Tensor(U), &block : U -> V) forall U, V
  strided_iteration(a) do |i|
    i.value = U.new(yield i.value)
  end
end
