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

require "../tensor"

# :nodoc:
struct Num::Internal::AxisIter(T, V)
  include Iterator(T)
  @shape : Array(Int32)
  @strides : Array(Int32)
  @inc : Int32
  @ptr : Pointer(V)
  @tmp : T
  @total : Int32
  @yielded : Int32 = 0
  @axis : Int32

  def initialize(arr : T, @axis : Int32 = -1, keepdims = false)
    if @axis < 0
      @axis += arr.rank
    end
    unless @axis < arr.rank
      raise Num::Internal::AxisError.new("Axis out of range for array")
    end

    @shape = arr.shape.dup
    @strides = arr.strides.dup
    @ptr = arr.to_unsafe
    @inc = arr.strides[axis]

    if keepdims
      @shape[axis] = 1
      @strides[axis] = 0
    else
      @shape.delete_at(axis)
      @strides.delete_at(axis)
    end

    @tmp = arr.class.new(@ptr, @shape, @strides)

    @total = arr.shape[axis]
  end

  def next
    if @yielded >= @total
      stop
    else
      ret = @tmp
      @yielded += 1
      @ptr += @inc
      @tmp = Tensor.new(@ptr, @shape, @strides)
      ret
    end
  end
end

# :nodoc:
struct Num::Internal::UnsafeAxisIter(T)
  include Iterator(T)
  @shape : Array(Int32)
  @strides : Array(Int32)
  @inc : Int32
  @ptr : Pointer(T)
  @tmp : Tensor(T)
  @total : Int32
  @yielded : Int32 = 0
  @axis : Int32

  def initialize(arr : Tensor(T), @axis : Int32 = -1, keepdims = false)
    if @axis < 0
      @axis += arr.rank
    end
    unless @axis < arr.rank
      raise Num::Internal::AxisError.new("Axis out of range for array")
    end

    @shape = arr.shape.dup
    @strides = arr.strides.dup
    @ptr = arr.to_unsafe
    @inc = arr.strides[axis]

    if keepdims
      @shape[axis] = 1
      @strides[axis] = 0
    else
      @shape.delete_at(axis)
      @strides.delete_at(axis)
    end

    @tmp = arr.class.new(@ptr, @shape, @strides)

    @total = arr.shape[axis]
  end

  def next
    ret = @tmp
    @ptr += @inc
    @tmp = Tensor.new(@ptr, @shape, @strides)
    ret
  end
end

# :nodoc:
module Num::Internal
  macro iter_macro(n, vars)
    struct ContigFlatIter{{n}}({% for v in vars %}{{v[:typ]}},{% end %})
      include Iterator(Tuple({% for v in vars %}{{v[:typ]}}, {% end %}))

      @size : Int32
      @offset : Int32

      {% for v in vars %}
        # tracks a single pointer to a single arrays data
        @ptr_{{v[:sym].id}} : Pointer({{v[:typ]}})
      {% end %}

      def initialize(
        {% for v in vars %}
          {{v[:sym].id}} : Tensor({{v[:typ]}}),
        {% end %}
      )

        {% for v in vars %}
          @ptr_{{v[:sym].id}} = {{v[:sym].id}}.to_unsafe
        {% end %}

        @offset = 0
        @size = {{vars[0][:sym].id}}.size
      end

      def next
        {% for v in vars %}
          ret_{{v[:sym].id}} = @ptr_{{v[:sym].id}}
        {% end %}

        if @offset < @size
          @offset += 1
          {% for v in vars %}
            @ptr_{{v[:sym].id}} += 1
          {% end %}
          {% if vars.size == 1 %}
            ret_{{vars[0][:sym].id}}
          {% else %}
            {
              {% for v in vars %}
                ret_{{v[:sym].id}},
              {% end %}
            }
          {% end %}
        else
          stop
        end
      end
    end
    # The primary N-dimensional iterator through any ndarrays.
    # this will always iterate in row-major order, regardless of the
    # underlying strides or memory layout of the array.
    struct NDFlatIter{{n}}({% for v in vars %}{{v[:typ]}},{% end %})
      include Iterator(Tuple({% for v in vars %}{{v[:typ]}}, {% end %}))

      # tracks the shape, dimensionality, and current iteration
      # of the ndarray, identifying if an ndarray iterator has finished
      # its iteration, as well as understanding the strides in each
      # dimension
      @shape : Pointer(Int32)
      @coord : Pointer(Int32)
      @dim : Int32

      {% for v in vars %}
        # tracks a single pointer to a single arrays data
        @ptr_{{v[:sym].id}} : Pointer({{v[:typ]}})
        @strides_{{v[:sym].id}} : Pointer(Int32)
        @backstrides_{{v[:sym].id}} : Pointer(Int32)
      {% end %}

      def initialize(
        {% for v in vars %}
          {{v[:sym].id}} : Tensor({{v[:typ]}}),
        {% end %}
      )
        @shape = {{vars[0][:sym].id}}.shape.to_unsafe
        @coord = Pointer(Int32).malloc({{vars[0][:sym].id}}.rank, 0)
        @dim = {{vars[0][:sym].id}}.rank - 1

        {% for v in vars %}
          @ptr_{{v[:sym].id}} = {{v[:sym].id}}.to_unsafe
          @strides_{{v[:sym].id}} = {{v[:sym].id}}.strides.to_unsafe
          @backstrides_{{v[:sym].id}} = Pointer(Int32).malloc(@dim+1) { |i| @strides_{{v[:sym].id}}[i] * (@shape[i] - 1) }
        {% end %}

        # If the strides of an array are negative, the pointer has to
        # be offset before iteration starts, otherwise the incorrect
        # elements will be yielded
        (@dim + 1).times do |i|
          {% for v in vars %}
            if @strides_{{v[:sym].id}}[i] < 0
              @ptr_{{v[:sym].id}} += (@shape[i] - 1) * @strides_{{v[:sym].id}}[i].abs
            end
          {% end %}
        end
        @done = false
      end

      # The step of the iterator, walks through each step of an ndarray in
      # row major order
      def next
        if @done
          return stop
        end

        # if the iterator has been exhausted, yield, which allows this iterator
        # to be zipped and used with each, map, etc.
        {% for v in vars %}
          ret_{{v[:sym].id}} = @ptr_{{v[:sym].id}}
        {% end %}

        @dim.step(to: 0, by: -1) do |k|
          if @coord[k] < @shape[k] - 1
            @coord[k] += 1
            {% for v in vars %}
              @ptr_{{v[:sym].id}} += @strides_{{v[:sym].id}}[k]
            {% end %}
            break
          else
            if k == 0
              @done = true
            end
            @coord[k] = 0
            {% for v in vars %}
              @ptr_{{v[:sym].id}} -= @backstrides_{{v[:sym].id}}[k]
            {% end %}
          end
        end
        {% if vars.size == 1 %}
          ret_{{vars[0][:sym].id}}
        {% else %}
          {
            {% for v in vars %}
              ret_{{v[:sym].id}},
            {% end %}
          }
        {% end %}
      end
    end
  end

  iter_macro 1, [{sym: :a, typ: T}]
  iter_macro 2, [{sym: :a, typ: T}, {sym: :b, typ: U}]
  iter_macro 3, [{sym: :a, typ: T}, {sym: :b, typ: U}, {sym: :c, typ: V}]
  iter_macro 4, [{sym: :a, typ: T}, {sym: :b, typ: U}, {sym: :c, typ: V}, {sym: :d, typ: W}]
end

# :nodoc:
struct Num::Internal::MatrixIter(T)
  include Iterator(T)
  getter tns : Tensor(T)
  getter axis : Int32
  getter dim : Int32
  getter ranges : Array(Int32 | Range(Int32, Int32))
  @idx : Int32 = 0

  def initialize(@tns : Tensor(T))
    if tns.rank < 3
      raise "Dimensionality of the Array is not high enough to reduce"
    end

    @axis = tns.rank - 3
    @dim = tns.shape[axis]
    @ranges = tns.shape.map_with_index do |a, i|
      axis == i ? 0 : 0...a
    end
  end

  def next
    if @idx == @dim
      return stop
    end
    @ranges[axis] = @idx
    @idx += 1
    tns[@ranges]
  end
end

struct Num::Internal::UnsafeNDFlatIter(T)
  include Iterator(T)
  @ptr : Pointer(T)
  @shape : Pointer(Int32)
  @strides : Pointer(Int32)
  @track : Pointer(Int32)
  @dim : Int32

  def initialize(arr : Tensor(T))
    @ptr = arr.to_unsafe
    @shape = arr.shape.to_unsafe
    @strides = arr.strides.to_unsafe
    @track = Pointer(Int32).malloc(arr.rank, 0)
    @dim = arr.rank - 1
    arr.rank.times do |i|
      if @strides[i] < 0
        @ptr += (@shape[i] - 1) * @strides[i].abs
      end
    end
  end

  def next
    ret = @ptr
    @dim.step(to: 0, by: -1) do |i|
      @track[i] += 1
      shape_i = @shape[i]
      stride_i = @strides[i]
      if @track[i] == shape_i
        @track[i] = 0
        @ptr -= (shape_i - 1) * stride_i
        next
      end
      @ptr += stride_i
      break
    end
    ret
  end
end
