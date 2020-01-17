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
require "../base"

macro iter_macro(n, vars)
  # The primary N-dimensional iterator through any ndarrays.
  # this will always iterate in row-major order, regardless of the
  # underlying strides or memory layout of the array.
  struct NumInternal::NDFlatIter{{n}}({% for v in vars %}{{v[:typ]}},{% end %})
    include Iterator(Tuple({% for v in vars %}{{v[:typ]}}, {% end %}))

    # tracks the shape, dimensionality, and current iteration
    # of the ndarray, identifying if an ndarray iterator has finished
    # its iteration, as well as understanding the strides in each
    # dimension
    @shape : Pointer(Int32)
    @track : Pointer(Int32)
    @dim : Int32

    {% for v in vars %}
      # tracks a single pointer to a single arrays data
      @ptr_{{v[:sym].id}} : Pointer({{v[:typ]}})
      @strides_{{v[:sym].id}} : Pointer(Int32)
    {% end %}

    def initialize(
      {% for v in vars %}
        {{v[:sym].id}} : Num::BaseArray({{v[:typ]}}),
      {% end %}
    )
      @shape = {{vars[0][:sym].id}}.shape.to_unsafe
      @track = Pointer(Int32).malloc({{vars[0][:sym].id}}.ndims, 0)
      @dim = {{vars[0][:sym].id}}.ndims - 1

      {% for v in vars %}
        @ptr_{{v[:sym].id}} = {{v[:sym].id}}.buffer
        @strides_{{v[:sym].id}} = {{v[:sym].id}}.strides.to_unsafe
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

      @dim.step(to: 0, by: -1) do |i|
        @track[i] += 1


        # Better to cache these here so they don't have to be continuously
        # looked up inside the inner loop
        shape_i = @shape[i]
        {% for v in vars %}
          stride_i{{v[:sym].id}} = @strides_{{v[:sym].id}}[i]
        {% end %}

        if @track[i] == shape_i
          if i == 0
            @done = true
          end
          @track[i] = 0
          {% for v in vars %}
            @ptr_{{v[:sym].id}} -= (shape_i - 1) * stride_i{{v[:sym].id}}
          {% end %}
          next
        end
        {% for v in vars %}
          @ptr_{{v[:sym].id}} += stride_i{{v[:sym].id}}
        {% end %}
        break
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
