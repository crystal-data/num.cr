require "./base"

module Bottle::Internal::ToString
  class BasePrinter(T)
    @t : BaseArray(T)
    @idx : Array(Int32)
    @io : IO
    @ptr : Pointer(T)
    getter obrackets : String
    getter cbrackets : String
    getter indent : String
    getter last_comma : Int32
    getter strides : Array(Int32)
    getter shape : Array(Int32)
    getter maxval : Int32

    def initialize(@t : BaseArray(T), @io, @prefix : String = "Base", @maxval = 5)
      @ptr = @t.@buffer
      @obrackets = prefix + "(" + "[" * @t.ndims
      @idx = [0] * @t.ndims
      @cbrackets = "]" * @t.ndims
      @indent = " " * @t.ndims
      @last_comma = 0
      @strides = @t.strides.dup
      @shape = @t.@shape.dup

      if t.@ndims > 2
        0.step(to: (@t.ndims // 2 - 1)) do |i|
          offset = @t.ndims - i - 1
          tmp = @strides[i]
          @strides[i] = @strides[offset]
          @strides[offset] = tmp

          tmp = @shape[i]
          @shape[i] = @shape[offset]
          @shape[offset] = tmp
        end
      elsif t.@ndims == 2
        @strides = @strides.reverse
        @shape = @shape.reverse
      end
      @io << @obrackets
    end

    def calc_ptr(idx)
      ret = @t.@buffer
      idx.zip(strides) do |i, j|
        ret += i * j
      end
      ret
    end

    def handle_value(value)
      {% if T == Float64 || T == Float32 %}
        value.round(3)
      {% elsif T == Complex %}
        "#{value.real.round(3)} + #{value.imag.round(3)}"
      {% else %}
        value
      {% end %}
    end

    def print
      if @t.ndims == 0
        @io << "[])"
      else
        @io << "#{handle_value(@ptr.value)}".rjust(maxval)
        until !inc
        end
        @io << ")"
      end
    end

    def inc
      first_item = 0
      ii = 0
      @idx[ii] += 1
      @ptr = calc_ptr(@idx)
      while (@idx[ii] == shape[ii])
        @idx[ii] = 0
        ii += 1
        if (ii == @idx.size)
          @io << "]" * ii
          return false
        end
        first_item += 1
        @idx[ii] += 1
        @ptr = calc_ptr(@idx)
      end

      if (ii != 0)
        @io << "]" * ii
        @io << ","
        @io << "\n " * ii
        @io << " " * (@prefix.size + 1) << indent[0...@t.ndims - ii - 1]
      end
      if first_item > 0
        @io << "[" * first_item
      else
        @io << ", "
      end
      @io << "#{handle_value(@ptr.value)}".rjust(maxval)
      true
    end
  end
end
