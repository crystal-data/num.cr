# Copyright (c) 2021 Crystal Data Contributors
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

module Num::Internal
  protected def offset_for_index(arr : Tensor, args : Array)
    new_shape = arr.shape.dup
    new_strides = arr.strides.dup

    acc = args.map_with_index do |arg, i|
      s_i, st_i, o_i = normalize(arr, arg, i)
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

    offset = arr.offset

    arr.rank.times do |k|
      if arr.strides[k] < 0
        offset += (arr.shape[k] - 1) * arr.strides[k].abs
      end
    end

    acc.zip(arr.strides) do |a, j|
      offset += a * j
    end

    {offset, new_shape, new_strides}
  end

  private def normalize(arr : Tensor, arg : Int, i : Int32)
    if arg < 0
      arg += arr.shape[i]
    end
    if arg < 0 || arg >= arr.shape[i]
      raise Num::Exceptions::IndexError.new(
        "Index #{arg} out of range for axis #{i} with size #{arr.shape[i]}"
      )
    end
    {0, 0, arg.to_i}
  end

  private def normalize(arr : Tensor, arg : Range, i : Int32)
    a_end = arg.end
    if a_end.is_a?(Int32)
      if a_end > arr.shape[i]
        arg = arg.begin...arr.shape[i]
      end
    end
    s, o = Indexable.range_to_index_and_count(arg, arr.shape[i]).not_nil!
    if s >= arr.shape[i]
      raise Num::Exceptions::IndexError.new(
        "Index #{arg} out of range for axis #{i} with size #{arr.shape[i]}"
      )
    end
    {o.to_i, arr.strides[i], s.to_i}
  end

  private def normalize(arr : Tensor, arg : Tuple(Range(B, E), Int), i : Int32) forall B, E
    range, step = arg
    abs_step = step.abs
    start, offset = Indexable.range_to_index_and_count(range, arr.shape[i]).not_nil!
    if start >= arr.shape[i]
      raise Num::Exceptions::IndexError.new(
        "Index #{arg} out of range for axis #{i} with size #{arr.shape[i]}"
      )
    end
    {offset // abs_step + offset % abs_step, step * arr.strides[i], start}
  end
end
