require "../api"

module Bottle::Internal
  macro iters(prefix, safe)
    struct {{prefix}}Flat(T)
      include Iterator(T)
      @ptr : Pointer(T)
      @size : Int32
      @step : Int32
      @offset : Int32

      def initialize(@ptr : Pointer(T), @size, @step)
        @offset = 0
      end

      def next
        ret = @ptr
        {% if safe %}
          if @offset < @size
            @offset += 1
            @ptr += @step
            ret
          else
            stop
          end
        {% else %}
          @ptr += @step
          ret
        {% end %}
      end
    end

    struct {{prefix}}ND(T)
      include Iterator(T)
      @ptr : Pointer(T)
      @shape : Pointer(Int32)
      @strides : Pointer(Int32)
      @track : Pointer(Int32)
      @dim : Int32

      def initialize(@ptr : Pointer(T), shape, strides, ndims)
        @shape = shape.to_unsafe
        @strides = strides.to_unsafe
        @track = Pointer(Int32).malloc(ndims, 0)
        @dim = ndims - 1
      end

      def next
        {% if safe %}
          if @done
            return stop
          end
        {% end %}

        ret = @ptr
        @dim.step(to: 0, by: -1) do |i|
          @track[i] += 1
          shape_i = @shape[i]
          stride_i = @strides[i]

          if @track[i] == shape_i
            {% if safe %}
              if i == 0
                @done = true
              end
            {% end %}
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

  end

  iters(Safe, true)
  iters(Unsafe, false)
end
