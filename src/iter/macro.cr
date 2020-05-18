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
require "../array/array"

module NumInternal
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
          {{v[:sym].id}} : AnyArray({{v[:typ]}}),
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
          {
            {% for v in vars %}
              ret_{{v[:sym].id}},
            {% end %}
          }
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
          {{v[:sym].id}} : AnyArray({{v[:typ]}}),
        {% end %}
      )
        @shape = {{vars[0][:sym].id}}.shape.to_unsafe
        @coord = Pointer(Int32).malloc({{vars[0][:sym].id}}.ndims, 0)
        @dim = {{vars[0][:sym].id}}.ndims - 1

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
        {
          {% for v in vars %}
            ret_{{v[:sym].id}},
          {% end %}
        }
      end
    end
  end

  iter_macro 2, [{sym: :a, typ: T}, {sym: :b, typ: U}]
  iter_macro 3, [{sym: :a, typ: T}, {sym: :b, typ: U}, {sym: :c, typ: V}]
  iter_macro 4, [{sym: :a, typ: T}, {sym: :b, typ: U}, {sym: :c, typ: V}, {sym: :d, typ: W}]
end
