********
Indexing
********

All indexing operations on Tensors return Tensors.  Even operations that in
other libraries might return scalars return zero dimensional Tensors in order
to avoid return-type ambiguity.  Indexing on Tensors also always return
views of the underlying data buffer of a tensor.  Modifying a view will
also modify the parent Tensor.

Slicing an axis
---------------

Tensors be indexed using Crystal's range objects to represent a slice along
a given axis.

.. code-block:: crystal

    t = Tensor.new([3, 3]) { |i| i }
    puts t[1...]

.. code-block:: crystal

    Tensor([[3, 4, 5],
            [6, 7, 8]])

Viewing an entry along an axis
------------------------------


If an integer is passed to an axis instead of a range, a view of the array
along that axis will be returned.

.. code-block:: crystal

    puts t[1]

.. code-block:: crystal

    Tensor([3, 4, 5])

Viewing vs. Slicing
-------------------

Using a range will *sometimes* remove a dimension from the output, while using an
integer will *always* remove a dimension from the returned tensor.  If a range
is provided where the first and last number are the same, it will be reduced
to an integer and the axis will be removed, otherwise the axis will remain.

.. code-block:: crystal

    puts t[1...1]
    puts t[1]
    puts t[1...2]

.. code-block:: crystal

    Tensor([3, 4, 5])
    Tensor([3, 4, 5])
    Tensor([[3, 4, 5]])

Accessing a Scalar
------------------

Since all indexing operations return scalars, accessing a scalar is done
through a method on a tensor called `value`, which returns the value of
a tensors data buffer.

.. code-block:: crystal

    puts t[2, 2]
    puts t[2, 2].value

.. code-block:: crystal

    Tensor(8)
    8

Strided indexing
----------------

By default, the step along each axis of a Tensor is 1, but this can
be manipulated by indexing operations.  In order to specify a step, pass
a tuple that includes a range, and a step.

.. code-block:: crystal

    puts t[{..., -1}]
    puts t[{..., 2}, {..., -1}]

.. code-block:: crystal

    Tensor([[6, 7, 8],
            [3, 4, 5],
            [0, 1, 2]])
    Tensor([[2, 1, 0],
            [8, 7, 6]])

A single indexing operation can be provided to each axis of a Tensor, allowing
for complex slicing and viewing of a tensor.
