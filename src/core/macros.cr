require "./converters"
require "./exceptions"
require "../base/base"
require "../tensor/tensor"

module Bottle::Internal
  include Convert
  # Broadcast two tensors against each other.  This will possibly
  # change the shape and strides of both passed tensors, so this
  # cannot be used on in-place operations.  To broadcast the inputs
  # of an operation to be conducted in place, use the ``broadcast_rhs``
  # macro instead.
  #
  # Raises ShapeError if the two shapes cannot be broadcast against
  # each other
  macro broadcast(a, b)
    if {{a}}.shape != {{b}}.shape
      bshape = {{a}}.broadcastable({{b}})
      if bshape.size >= 1
        {{a}} = {{a}}.broadcast_to(bshape)
        {{b}} = {{b}}.broadcast_to(bshape)
      end
    end
  end

  # Broadcasts the right-hand-size argument against the left-hand-side
  # argument passed.  This is helpful if an operation needs to be done in place,
  # but the other operator needs to be broadcasted.
  #
  # Raises ShapeError if the two shapes cannot be broadcast against
  # each other
  macro broadcast_rhs(a, b)
    if {{a}}.shape != {{b}}.shape
      {{b}} = {{b}}.broadcast_to({{a}}.shape)
    end
  end

  # Takes in numerical arguments, arrays, numbers, anything that can be
  # converted to a Tensor, and coerces the input to a Tensor.  This will
  # *not* make copies of underlying data, unless the data has to be copied,
  # such as creating a Tensor from an array or scalar.
  #
  # This method will not maintain subclasses past through, useful for coercing
  # subclasses of BaseArray to Tensors as well.
  macro upcast_if(*args)
    {% for arg in args %}
      {{arg}} = astensor({{arg}})
    {% end %}
  end

  macro clipaxis(axis, size)
    if {{axis}} < 0
      {{axis}} += {{size}}
    end
    if {{axis}} < 0 || {{axis}} > {{size}}
      raise "Axis out of range"
    end
  end

  # Returns an outer iterator across two n-dimensional tensors.  This loops
  # over the elements of each tensor in order, returning the outer
  # relationships.  This is used to add outer functionality to any
  # particular function.
  private def outiter(a, b)
    index = 0
    a.flat_iter.each do |i|
      b.flat_iter.each do |j|
        yield i, j, index
        index += 1
      end
    end
  end

  # Defines an operation to be applied elementwise to a Tensor.
  # Inputs will be broadcast against each other, and type coercing will
  # happen if the operation is in place, otherwise the type will be
  # inferred.
  #
  # Also extends a function to support an outer operation, operating
  # across all combinations of elements of two arrays.
  macro elementwise(operator, name)
    def {{name}}(a : Tensor, b : Tensor)
      broadcast a, b
      itera = a.unsafe_iter
      iterb = b.unsafe_iter

      Tensor.new(a.shape) do |_|
        itera.next.value {{operator.id}} iterb.next.value
      end
    end

    def {{name}}(a : Tensor, b : Number)
      itera = a.unsafe_iter
      Tensor.new(a.shape) do |_|
        itera.next.value {{operator.id}} b
      end
    end

    def {{name}}(a : Number, b : Tensor)
      iterb = b.unsafe_iter
      Tensor.new(a.shape) do |_|
        a {{operator.id}} iterb.next.value
      end
    end

    def {{name}}!(a : Tensor(U), b) forall U
      upcast_if b
      broadcast_rhs a, b
      a.flat_iter.zip(b.flat_iter) do |i, j|
        i.value = U.new(i.value {{operator.id}} j.value)
      end
    end

    def {{name}}_outer(a : Tensor, b : Tensor)
      ret = Tensor(typeof(a.value {{operator.id}} b.value)).new(a.shape + b.shape)
      buf = ret.buffer
      outer(a, b) do |i, j, idx|
        buf[idx] = i.value {{operator.id}} j.value
      end
    end
  end

  # Wraps a function from the standard library to be applied elementwise
  # to a single Tensor.
  macro stdlibwrap(func)
    def {{func}}(a, b)
      upcast_if a, b
      broadcast a, b
      itera = a.unsafe_iter
      iterb = b.unsafe_iter

      Tensor.new(a.shape) do |_|
        Math.{{func}}(itera.next.value, iterb.next.value)
      end
    end

    def {{func}}!(a : Tensor(U), b) forall U
      upcast_if b
      broadcast_rhs a, b
      a.flat_iter.zip(b.flat_iter) do |i, j|
        i.value = U.new(Math.{{func}}(i.value, j.value))
      end
    end

    def {{func}}_outer(a : Tensor, b : Tensor)
      ret = Tensor(typeof(Math.{{func}}(a.value, b.value))).new(a.shape + b.shape)
      buf = ret.buffer
      outer(a, b) do |i, j, idx|
        buf[idx] = Math.{{func}}(i.value, j.value)
      end
    end
  end

  # Wraps a function from the standard library that takes more than one
  # tensor as an argument.  All of these functions will upcast scalars
  # to tensors.
  macro stdlibwrap1d(func)
    def {{func}}(a)
      upcast_if a
      iter = a.unsafe_iter
      Tensor.new(a.shape) do |_|
        Math.{{func}}(iter.next.value)
      end
    end
  end

  macro reducescalar(operator, initial, arg)
    {{arg}}.flat_iter.reduce(U.new({{initial}})) { |i, j| i {{operator.id}} j.value }
  end

  macro reducebool(operator, initial, arg)
    {{arg}}.flat_iter.reduce({{initial}}) { |i, j| i {{operator.id}} j.value }
  end

  macro reduceaxis(operator, arg)
    {{arg}}.reduce_fast(axis, keepdims) do |i, j|
      i.value {{operator.id}}= j.value
    end
  end
end
