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

require "../api"
require "./series"
require "./frame_slice"
require "csv"

class DataFrame(T, V)
  getter c : T
  getter index : Index(V)
  getter size : Int32

  # Create a `DataFrame` from an index, and a NamedTuple
  # of values.  This is private so that all other methods
  # can coerce the values of `data` to `Series`, so that
  # no other types can be present in a `DataFrame`
  #
  # Arguments
  # ---------
  # `index` : Index(V)
  #   Index for all Series in the DataFrame
  # `data` : NamedTuple
  #   Schema and Series of the DataFrame
  def initialize(index : Index(V), size : Int, **data : **T)
    @c = data
    @index = index
    @size = size.to_i
  end

  # Create a `DataFrame` from a variadic number of arguments.
  #
  # The arguments can be of a flexible type, as long as they are
  # one dimensional and can be cast to `Tensor`s, and therefore
  # `Series`.
  #
  # Arguments
  # ---------
  # `data` : NamedTuple of Tensor's or Enumerables
  #   Data to convert into a `DataFrame`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.random(0.0...10.0, [4])
  # b = [1, 2, 3, 4]
  #
  # df = DataFrame.from_items(a: a, b: b)
  # puts df
  #
  #          a  b
  # 0  4.06169  1
  # 1  7.55353  2
  # 2  1.26119  3
  # 3  1.16003  4
  # ```
  def self.from_items(**data : **U) forall U
    {% begin %}
      data = NamedTuple.new(
      {% for key, value in U %}
        {{key}}: Series.new(
            data[{{key.symbolize}}].to_tensor,
            name: {{key.symbolize}}
          ),
      {% end %}
      )
    {% end %}
    size = self.check_attributes(data)
    ii = Index.range(size)
    data.each do |k, v|
      v.set_index!(ii)
    end
    new(ii, size, **data)
  end

  # :nodoc:
  private def self.check_attributes(data) : Int32
    s0 = 0
    data.each_with_index do |k, v, i|
      if i == 0
        s0 = v.size
      end

      if v.size != s0
        raise Num::Internal::ShapeError.new("All inputs must be the same size")
      end
    end
    s0
  end

  def [](i : V)
    {% begin %}
      data = NamedTuple.new(
        {% for key, value in T %}
          {{key}}: @c[{{key.symbolize}}][i],
        {% end %}
      )
      FrameSlice.new(**data)
    {% end %}
  end

  # :nodoc:
  def each
    @size.times do |i|
      {% begin %}
        data = NamedTuple.new(
        {% for key, value in T %}
          {{key}}: @c[{{key.symbolize}}].iat(i),
        {% end %}
        )
        yield data
      {% end %}
    end
  end

  def each_with_index
    i = 0
    each do |e|
      yield e, @index.iat(i).key
      i += 1
    end
  end

  # :nodoc:
  def to_s(io)
    iw = @index.max_repr_width
    dw = Hash(Symbol, Int32).new
    @c.each do |k, v|
      dw[k] = {"#{k}".size, v.max_repr_width}.max
    end
    io << " ".rjust(iw)
    io << "  "
    @c.each do |k, v|
      io << "#{k}".rjust(dw[k])
      io << "  "
    end
    io << "\n"
    i = 0
    each_with_index do |e, i|
      io << "#{Num::Internal.format(i)}".rjust(iw)
      io << "  "
      e.each do |k, v|
        if k != :index
          io << "#{Num::Internal.format(v)}".rjust(dw[k])
          io << "  "
        end
      end
      io << "\n"
    end
  end

  # :nodoc:
  def to_csv
    CSV.build do |csv|
      csv.row do |r|
        r << "index"
        @c.each_key do |k|
          r << k
        end
      end
      each_with_index do |e, i|
        csv.row do |r|
          r << i
          e.each_value do |v|
            r << v
          end
        end
      end
    end
  end

  # :nodoc:
  macro reduce(reduction)
    def {{reduction.id}}
      \{% begin %}
        FrameSlice.new(
          \{% for key, value in T %}
            \{{key}}: @c[\{{key.symbolize}}].{{reduction.id}},
          \{% end %}
        )
      \{% end %}
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
  reduce unique
  reduce to_a

  # :nodoc:
  macro elementwise(fn)
    def {{fn.id}}(other)
      \{% begin %}
        data = NamedTuple.new(
          \{% for key, value in T %}
            \{{key}}: @c[\{{key.symbolize}}].{{fn.id}}(other),
          \{% end %}
        )
        DataFrame.new(@index, @size, **data)
      \{% end %}
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

macro agg(nt, **fns)
  FrameSlice.new(
    {% for k, v in fns %}
        {{k}}: {{nt}}.c[{{k.symbolize}}].{{v.id}},
    {% end %}
  )
end
