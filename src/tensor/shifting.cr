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

class Tensor(T)
  # Broadcasts a `Tensor` to a new shape.  Returns a read-only
  # view of the original `Tensor`.  Many elements in the `Tensor`
  # will refer to the same memory location, and the result is
  # rarely contiguous.
  #
  # Shapes must be broadcastable, and an error will be raised
  # if broadcasting fails.
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   The shape of the desired output `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.broadcast_to([3, 3])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  def as_shape(shape : Array(Int)) : Tensor(T)
    Num::Internal.broadcast_to(@storage, shape.map &.to_i)
  end

  # Casts a `Tensor` to a new dtype, by making a copy.  Information may
  # be lost when converting between data types, for example Float to Int
  # or Int to Bool.
  #
  # Arguments
  # ---------
  # *u* : U.class
  #   Data type the `Tensor` will be cast to
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.5, 3.5]
  #
  # a.astype(Int32)   # => [1, 2, 3]
  # a.astype(Bool)    # => [true, true, true]
  # a.astype(Float32) # => [1.5, 2.5, 3.5]
  # ```
  def as_type(u : U.class) : Tensor(U) forall U
    Num::Backend.cast_tensor(@storage, U)
  end

  # Deep-copies a `Tensor`.  If an order is provided, the returned
  # `Tensor`'s memory layout will respect that order.
  #
  # If no order is provided, the `Tensor` will retain it's same
  # memory layout.
  #
  # Arguments
  # ---------
  # *order* : Num::OrderType?
  #   Memory layout to use for the returned `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.dup # => [1, 2, 3]
  # ```
  def dup(order : Num::OrderType? = nil) : Tensor(T)
    if order.nil?
      if self.is_f_contiguous
        order = Num::ColMajor
      else
        order = Num::RowMajor
      end
    end
    Num::Backend.copy_tensor(@storage, order)
  end

  # Permutes a `Tensor`'s axes to a different order.  This will
  # always create a view of the permuted `Tensor`.
  #
  # Arguments
  # ---------
  # *axes* : Array(Int)
  #   New ordering of axes for the permuted `Tensor`.  If empty,
  #   a full transpose will occur
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([4, 3, 2]) { |i| i }
  # a.transpose([2, 0, 1])
  #
  # # [[[ 0,  2,  4],
  # #   [ 6,  8, 10],
  # #   [12, 14, 16],
  # #   [18, 20, 22]],
  # #
  # #  [[ 1,  3,  5],
  # #   [ 7,  9, 11],
  # #   [13, 15, 17],
  # #   [19, 21, 23]]]
  # ```
  def transpose(axes : Array(Int) = [] of Int32)
    order = axes.map &.to_i
    new_shape = self.shape.dup
    new_strides = self.strides.dup

    if order.size == 0
      order = (0...self.rank).to_a.reverse
    end

    n = order.size
    if n != self.rank
      raise "Axes do not match Tensor"
    end

    perm = [0] * self.rank
    r_perm = [-1] * self.rank

    n.times do |i|
      axis = order[i]
      if axis < 0
        axis = self.rank + axis
      end
      if axis < 0 || axis >= self.rank
        raise "Invalid axis for Tensor"
      end
      if r_perm[axis] != -1
        raise "Repeated axis in transpose"
      end
      r_perm[axis] = i
      perm[i] = axis
    end

    n.times do |i|
      new_shape[i] = shape[perm[i]]
      new_strides[i] = strides[perm[i]]
    end

    new_storage = @storage.class.new(@storage.data, new_shape, new_strides, @storage.size, @storage.offset)
    Tensor(T).new(new_storage)
  end
end
