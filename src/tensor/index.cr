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
  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def [](*args) : Tensor(T)
    slice(args.to_a)
  end

  # :ditto:
  def [](args : Array) : Tensor(T)
    slice(args)
  end

  # The primary method of setting Tensor values.  The slicing behavior
  # for this method is identical to the `[]` method.
  #
  # If a `Tensor` is passed as the value to set, it will be broadcast
  # to the shape of the slice if possible.  If a scalar is passed, it will
  # be tiled across the slice.
  #
  # Arguments
  # ---------
  # *args* : *U
  #   Tuple of arguments.  All but the last argument must be valid
  #   indexer, so a `Range`, `Int`, or `Tuple(Range, Int)`.  The final
  #   argument passed is used to set the values of the `Tensor`.  It can
  #   be either a `Tensor`, or a scalar value.
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def []=(*args : *U) forall U
    {% begin %}
       set(
         {% for i in 0...U.size - 1 %}
           args[{{i}}],
         {% end %}
         value: args[{{U.size - 1}}]
       )
     {% end %}
  end

  # :ditto:
  def []=(args : Array, value)
    set(args, value)
  end

  # :nodoc:
  def value : T
    @storage.value
  end

  private def slice(args : Array)
    new_shape = self.shape.dup
    new_strides = self.strides.dup

    acc = args.map_with_index do |arg, i|
      s_i, st_i, o_i = normalize(arg, i)
      new_shape[i] = s_i
      new_strides[i] = st_i
      o_i
    end

    i = 0
    new_strides.reject! do
      condition = new_shape[i] == 0
      i += 1
      condition
    end

    new_shape.reject! do |j|
      j == 0
    end

    offset = @storage.offset

    rank.times do |k|
      if self.strides[k] < 0
        offset += (self.shape[k] - 1) * self.strides[k].abs
      end
    end

    acc.zip(self.strides) do |a, j|
      offset += a * j
    end

    Num::Backend.slice_storage_by_offset(@storage, new_shape, new_strides, new_shape.product, offset)
  end

  private def normalize(arg : Int, i : Int32)
    if arg < 0
      arg += self.shape[i]
    end
    if arg < 0 || arg >= self.shape[i]
      raise "Index #{arg} out of range for axis #{i} with size #{self.shape[i]}"
    end
    {0, 0, arg.to_i}
  end

  private def normalize(arg : Range, i : Int32)
    a_end = arg.end
    if a_end.is_a?(Int32)
      if a_end > self.shape[i]
        arg = arg.begin...self.shape[i]
      end
    end
    s, o = Indexable.range_to_index_and_count(arg, self.shape[i])
    if s >= self.shape[i]
      raise "Index #{arg} out of range for axis #{i} with size #{self.shape[i]}"
    end
    {o.to_i, self.strides[i], s.to_i}
  end

  private def normalize(arg : Tuple(Range(B, E), Int), i : Int32) forall B, E
    range, step = arg
    abs_step = step.abs
    start, offset = Indexable.range_to_index_and_count(range, self.shape[i])
    if start >= self.shape[i]
      raise "Index #{arg} out of range for axis #{i} with size #{self.shape[i]}"
    end
    {offset // abs_step + offset % abs_step, step * self.strides[i], start}
  end

  private def set(mask : Tensor(Bool), value : Number)
    m = mask.as_shape(@shape)
    map!(m) do |i, c|
      c ? value : i
    end
  end

  private def set(*args, value)
    set(args.to_a, value)
  end

  private def set(args : Array, t : Tensor)
    s = self[args]
    t = t.as_shape(s.shape)
    if t.rank > s.rank
      raise "Setting a Tensor with a sequence"
    end
    s.map!(t) do |_, j|
      j
    end
  end

  private def set(args : Array, t : U) forall U
    s = self[args]
    s.map! do
      T.new(t)
    end
  end
end
