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

require "./index"

# A `Series` is a one dimensional view of a `Frame`.
# Slicing column-wise will return a `Series`.  A `Series`
# stores its own index information even if an index
# is being tracked by a Frame.
class Series(T, V)
  getter data : Tensor(T)
  getter name : Symbol
  getter index : Index(V)

  # Initializes a `Series` from its components.  This
  # method is only used by internal calls
  #
  # Arguments
  # `data` : Tensor(T)
  #   Tensor containing the data of a series
  # `name` : Symbol
  #   Series description
  # `index`
  #   Hash allowing fast access to `Series` values
  private def initialize(data : Tensor(T), name : Symbol, index : Index(V))
    @data = data
    @name = name
    @index = index
  end

  # Initializes a `Series` from a `Tensor` and a `name`.
  # If no name is provided the `Series` will be unnamed,
  # and a name will be inferred if adding this to a
  # `Frame`
  #
  # Arguments
  # ---------
  # `t` : Tensor(U)
  #   One dimensional input `Tensor`
  # `name` : Symbol
  #   Identifier for the `Series`
  def self.new(t : Tensor(U), name : Symbol = :unnamed) forall U
    index = Index.range(t.size)
    new(t, name, index)
  end

  # Initializes a `Series` from a `Tensor`, `Index` and a `name`.
  # If no name is provided the `Series` will be unnamed,
  # and a name will be inferred if adding this to a
  # `Frame`
  #
  # Arguments
  # ---------
  # `t` : Tensor(U)
  #   One dimensional input `Tensor`
  # `index` : Index(V)
  #   Index for the `Series`
  # `name` : Symbol
  #   Identifier for the `Series`
  def self.new(
    t : Tensor(U),
    index : Index(V),
    name : Symbol = :unnamed
  ) forall U, V
    new(t, name, index)
  end

  # :nodoc:
  def each
    @data.each do |e|
      yield e
    end
  end

  # :nodoc:
  def each_with_index
    @index.each do |k, v|
      yield @data[v].value, k
    end
  end

  def [](i : V)
    row = @index[i]
    @data[row].value
  end

  # :nodoc:
  def iat(i : Int)
    @data[i].value
  end

  # :nodoc:
  def size
    @data.size
  end

  def set_index!(index : Index(V))
    @index = index
  end

  # :nodoc:
  def to_s(io)
    iw = @index.max_repr_width
    vw = self.max_repr_width
    each_with_index do |e, i|
      io << "#{Num::Internal.format(i)}".ljust(iw)
      io << "  "
      io << "#{Num::Internal.format(e)}".rjust(vw)
      io << "\n"
    end
    io << "Name: #{@name}\n"
    io << "dtype: #{T}"
  end

  # :nodoc:
  def max_repr_width
    w = 0
    each do |k|
      l = Num::Internal.format(k).size
      if l > w
        w = l
      end
    end
    w
  end

  private macro reduce(fn)
    def {{fn.id}}
      Num.{{fn.id}}(@data)
    end
  end

  reduce sum
  reduce prod
  reduce min
  reduce max
  reduce mean
  reduce std
  reduce all
  reduce any

  private macro reduce_on(fn)
    def {{fn.id}}
      @data.{{fn.id}}
    end
  end

  reduce_on unique
  reduce_on to_a
  reduce_on to_tensor

  private macro elementwise(fn)
    def {{fn.id}}(other : Number)
      t = Num.{{fn.id}}(@data, other)
      Series.new(t, @index, @name)
    end

    def {{fn.id}}(other : FrameSlice)
      t = Num.{{fn.id}}(@data, other.c[@name])
      Series.new(t, @index, @name)
    end
  end

  elementwise add
  elementwise subtract
  elementwise multiply
  elementwise divide
  elementwise floordiv
  elementwise power
  elementwise modulo
  elementwise left_shift
  elementwise right_shift
  elementwise bitwise_and
  elementwise bitwise_or
  elementwise bitwise_xor
  elementwise equal
  elementwise not_equal
  elementwise greater
  elementwise greater_equal
  elementwise less
  elementwise less_equal
end
