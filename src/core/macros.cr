require "../base/base"
require "../tensor/tensor"
require "./converters"

module Bottle::Internal::Core
  # Broadcast two tensors against each other.  This will possibly
  # change the shape and strides of both passed tensors, so this
  # cannot be used on in-place operations.  To broadcast the inputs
  # of an operation to be conducted in place, use the ``broadcast_rhs``
  # macro instead.
  #
  # Raises ShapeError if the two shapes cannot be broadcast against
  # each other
  macro broadcast(a, b)
    bshape = {{a}}.broadcastable({{b}})
    if bshape.size > 1
      {{a}} = {{a}}.broadcast_to(bshape)
      {{b}} = {{b}}.broadcast_to(bshape)
    end
  end

  # Broadcasts the right-hand-size argument against the left-hand-side
  # argument passed.  This is helpful if an operation needs to be done in place,
  # but the other operator needs to be broadcasted.
  #
  # Raises ShapeError if the two shapes cannot be broadcast against
  # each other
  macro broadcast_rhs(a, b)
    {{b}} = {{b}}.broadcast_to({{a}}.shape)
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

  private def outiter(a, b)
    index = 0
    a.flat_iter.each do |i|
      b.flat_iter.each do |j|
        yield i, j, index
        index += 1
      end
    end
  end

  macro elementwise(operator, name)
    def {{name}}(a, b) forall U, V
      upcast_if a, b
      broadcast a, b
      itera = a.unsafe_iter
      iterb = b.unsafe_iter

      a.basetype.new(a.shape) do |_|
        itera.next.value {{operator.id}} iterb.next.value
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
      typeout = typeof(a.value {{operator.id}} b.value)
      ret = Tensor(typeout).new(a.shape + b.shape)
      buf = ret.buffer
      outer(a, b) do |i, j, idx|
        buf[idx] = i.value {{operator.id}} j.value
      end
    end
  end
end
