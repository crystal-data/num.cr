# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
require "./array"
require "../base/constants"
require "../base/exceptions"

module NumInternal
  # Determines if two shapes are broadcastable against each other.
  # The rules for checking this property are well defined:
  #
  # Two dimensions are compatible if:
  #   - they are equal
  #   - one of them is equal to 1
  #
  # If the axes of the array are different lengths, dimensions of
  # size one can be appended to one or the other in order to make
  # the arrays broadcastable against each other and satisfy the
  # rules for broadcastable dimensions.
  def broadcastable(arr : AnyArray, other : AnyArray)
    # Fast track instances where the two shapes already match, no
    # need in pointlessly calculating the same shape that
    # already exists
    return [] of Int32 unless arr.shape != other.shape

    sz = arr.shape.size
    osz = other.shape.size

    # If the sizes already match, the rules are well defined
    # to make a broadcast.
    if sz == osz
      # Check the shapes, return the new shape, both arrays will
      # be broadcasted, so this can't be used for in-place operations,
      # only one of the arrays can be broadcasted in that case.
      if broadcast_equal(arr.shape, other.shape)
        return broadcastable_shape(arr.shape, other.shape)
      end
    else
      # Both of these paths prepend ones to the smaller shape
      # in order to match broadcasting rules.
      if sz > osz
        othershape = [1] * (sz - osz) + other.shape
        if broadcast_equal(arr.shape, othershape)
          return broadcastable_shape(arr.shape, othershape)
        end
      else
        selfshape = [1] * (osz - sz) + arr.shape
        if broadcast_equal(selfshape, other.shape)
          return broadcastable_shape(selfshape, other.shape)
        end
      end
    end
    # If no broadcasting is possible, raise a ShapeError.  No other
    # result makes sense, operation has to fail.
    raise ShapeError.new("Shapes #{arr.shape} and #{other.shape} are not broadcastable")
  end

  # Broadcasts an array to a new shape. A readonly view on the original array
  # with the given shape. It is typically not contiguous. Furthermore,
  # more than one element of a broadcasted array may refer to a single
  # memory location.
  def broadcast_to(arr : AnyArray, newshape : Array(Int32))
    dim = newshape.size
    defstrides = [0] * dim
    sz = 1
    dim.times do |i|
      defstrides[dim - i - 1] = sz
      sz *= newshape[dim - i - 1]
    end

    newstrides = broadcast_strides(newshape, arr.shape, defstrides, arr.strides)
    newflags = arr.flags.dup
    newflags &= ~Num::ArrayFlags::Write & ~Num::ArrayFlags::OwnData

    arr.class.new(arr.to_unsafe, newshape, newstrides, newflags)
  end

  macro broadcast_n(items)
    def broadcast({% for item in items %}{{item[:sym].id}} : AnyArray({{item[:typ]}}),{% end %}) forall {% for item, index in items %} {{item[:typ]}} {% if index != items.size - 1 %},{% end %} {% end %}
      t = {
        {% for item in items %} {{item[:sym].id}}, {% end %}
      }
      if t.all? { |i| i.shape == t[0].shape }
        return t
      end

      nd = t.max_of { |i| i.ndims }
      shape = [0] * nd

      t.size.times do |i|
        diff = nd - t[i].shape.size
        tshape = [1] * diff + t[i].shape
        shape = shape.map_with_index do |e, i|
          e > tshape[i] ? e : tshape[i]
        end
      end

      {
        {% for item in items %} bcast_if({{item[:sym].id}}, shape), {% end %}
      }
    end
  end

  def broadcast2(a : AnyArray(U), b : AnyArray(V)) forall U, V
    t = {a, b}
    if a.shape == b.shape
      return t
    end

    nd = t.max_of { |i| i.ndims }
    shape = [0] * nd

    2.times do |i|
      diff = nd - t[i].shape.size
      tshape = [1] * diff + t[i].shape
      shape = shape.map_with_index do |e, j|
        e > tshape[j] ? e : tshape[j]
      end
    end

    return {bcast_if(t[0], shape), bcast_if(t[1], shape)}
  end

  broadcast_n [{sym: :a, typ: T}, {sym: :b, typ: U}]
  broadcast_n [{sym: :a, typ: T}, {sym: :b, typ: U}, {sym: :c, typ: V}]
  broadcast_n [{sym: :a, typ: T}, {sym: :b, typ: U}, {sym: :c, typ: V}, {sym: :d, typ: W}]

  def bcast_if(item : AnyArray, shape : Array(Int32))
    shape == item.shape ? item : item.broadcast_to(shape)
  end

  # as_strided creates a view into the array given the exact strides and
  # shape. This means it manipulates the internal data structure of
  # a Tensor and, if done incorrectly, the array elements can point
  # to invalid memory and can corrupt results or crash your program.
  # It is advisable to always use the original x.strides when
  # calculating new strides to avoid reliance on a contiguous
  # memory layout.
  #
  # Furthermore, arrays created with this function often contain self
  # overlapping memory, so that two elements are identical.
  # Vectorized write operations on such arrays will typically be
  # unpredictable. They may even give different results for
  # small, large, or transposed arrays. Since writing to these
  # arrays has to be tested and done with great care, you may want
  # to use writeable=false to avoid accidental write operations.
  def as_strided(arr : AnyArray, shape : Array(Int32), strides : Array(Int32))
    newflags = arr.flags.dup
    newflags &= ~Num::ArrayFlags::Write & ~Num::ArrayFlags::OwnData
    arr.class.new(arr.to_unsafe, shape, strides, newflags)
  end

  # Finds the strides that must be present in order to broadcast an existing
  # array to a new array, or raises that the array cannot be broadcasted into
  # the provided shape.
  #
  # This method is primarily used by unsafe methods, such as `broadcast_to`.
  # When using this method, the resulting array's flags should ideally be
  # set to readonly, since many locations can share memory.  This method is
  # a safer alternative to `as_strided`
  private def broadcast_strides(dest_shape, src_shape, dest_strides, src_strides)
    # Find where the strides need to begin to match the input
    # shape/strides to the new shape
    dims = dest_shape.size
    start = dims - src_shape.size

    ret = [0] * dims
    (dims - 1).step(to: start, by: -1) do |i|
      s = src_shape[i - start]
      case s
      # Zero strides in a dimension is the easiest way to "trick"
      # the nditerator to traverse that dimension multiple times
      # and produce the same value.  This does however mean that
      # the iterator will produce many instances of the same pointer
      # when iterating through a broadcasted array.
      #
      # This is the reason for the read only flag on an array.
      when 1
        ret[i] = 0
      when dest_shape[i]
        # Otherwise the broadcasted strides will be computed from
        # the source strides, this path will always be chosen when
        # trying to broadcast to an identically shaped array for example.
        ret[i] = src_strides[i - start]
      else
        # Since the zero shaped dimensions will appear with invalid broadcasts,
        # raise here to indicate that the two shapes are incompatible.
        raise ShapeError.new("Cannot broadcast from #{src_shape} to #{dest_shape}")
      end
    end
    ret
  end

  # This method checks if two shapes are broadcastable with each other
  # in their current form.  There are several manipulations that can
  # be done on shapes to make them broadcastable eventually, so this
  # may be checked several times when broadcasting.
  private def broadcast_equal(a, b)
    bc = true
    a.zip(b) do |i, j|
      # Shapes can be broadcast against each other if for every dimension
      # the following is true: the dimensions are equal among the two shapes,
      # or either of the dimensions is equal to 1
      if !(i == j || i == 1 || j == 1)
        bc = false
      end
    end
    bc
  end

  # Once an array is determined to be broadcastable, the resulting
  # shape is simply the maximum value found at each dimension of the
  # two shapes.
  private def broadcastable_shape(a, b)
    a.zip(b).map do |i|
      Math.max(i[0], i[1])
    end
  end
end
