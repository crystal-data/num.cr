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

require "../tensor/tensor"
require "./internal/context"

# A `ClTensor` is a multidimensional container of fixed size, containing
# elements of type T.
#
# The number of dimensions is specified by a `ClTensor`'s `shape`, which
# is an `Array` of integers specifying the size of a `Tensor` in each
# dimension.
#
# A `ClTensor` can be created from a wide variety of creation methods.
# Including from a scalar value and `shape`.
#
# `ClTensor`'s store data using a `LibCL::ClMem`
#
# `ClTensor`'s cannot be resized, and any operation the changes the total
# number of elements in a `ClTensor` will return a new object.
class ClTensor(T)
  getter shape : Array(Int32)
  getter size : Int32

  # Creates a `ClTensor` from a provided shape, initializing
  # the `ClTensor`'s buffer without filling it with data
  #
  # When creating a `ClTensor` from a shape alone, the generic
  # type must be specified
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   Size of `ClTensor` in each dimension
  #
  # Examples
  # --------
  # ```
  # t = ClTensor(Float32).new([3, 3, 2])
  # ```
  def initialize(shape : Array(Int))
    check_type
    @shape = shape.map &.to_i
    @size = @shape.product
    @buffer = Cl.buffer(
      Num::ClContext.instance.context,
      UInt64.new(@size),
      dtype: T
    )
  end

  # Creates a `ClTensor` from a provided buffer and shape.
  # This should primarily be used by internal methods that
  # need to create `ClTensor`s or pass to lower level
  # libraries
  #
  # The generic type of the `ClTensor` must be specified, as
  # LibCL::ClMem does not store this information
  #
  # Arguments
  # ---------
  # *buffer* : LibCL::ClMem
  #   Memory buffer for a `ClTensor`
  # *shape* : Array(Int)
  #   Size of the `ClTensor` in each dimension
  #
  # Examples
  # --------
  # ```
  # t = ClTensor(Float32).new(buffer, [3, 3, 2])
  # ```
  def initialize(buffer : LibCL::ClMem, shape : Array(Int))
    check_type
    @buffer = buffer
    @shape = shape.map &.to_i
    @size = @shape.product
  end

  # Creates a `ClTensor` from a provided shape, initializing
  # the `ClTensor`'s buffer with a value specified by the
  # user
  #
  # The generic type of the `ClTensor` is inferred from the
  # provided value
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   Size of `ClTensor` in each dimension
  # *value* : T
  #   Value to fill the `ClTensor`
  #
  # Examples
  # --------
  # ```
  # t = ClTensor(Float32).new([3, 3, 2], 3.5)
  # ```
  def initialize(shape : Array(Int), value : T)
    check_type
    @shape = shape.map &.to_i
    @size = @shape.product
    @buffer = Cl.buffer(
      Num::ClContext.instance.context,
      @size.to_u64,
      dtype: T
    )
    Cl.fill(Num::ClContext.instance.queue, @buffer, value, @size.to_u64)
  end

  # Move a `ClTensor` from an OpenCL buffer to a CPU `Tensor` backed
  # by a `Pointer`.  This operation is slow, and should not be
  # used frequently.  Operations are most efficient when done exclusively
  # on a single architecture
  #
  # This method always blocks on write
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # t = ClTensor(Float32).new([5], 2.2)
  # t.cpu # => [2.2, 2.2, 2.2, 2.2, 2.2]
  # ```
  def cpu : Tensor(T)
    t = Tensor(T).new(@shape)
    LibCL.cl_enqueue_read_buffer(
      Num::ClContext.instance.queue,
      @buffer,
      LibCL::CL_TRUE,
      0_u64,
      (@size * sizeof(T)).to_u64,
      t.to_unsafe,
      0_u32, nil, nil
    )
    t
  end

  # :nodoc:
  def check_type
    {% unless T == Float32 || T == Float64 %}
      {% raise "Bad dtype: #{T}. #{T} is not supported for ClTensors" %}
    {% end %}
  end

  # :nodoc:
  def free
  end

  # :nodoc:
  def to_unsafe
    @buffer
  end

  # :nodoc:
  def rank
    @shape.size
  end
end
