***************
Tensor Creation
***************

Introduction
------------

There are several general ways to create n-dimensional tensors:

#. Conversion from a Crystal Array
#. Bottle tensor creation routines (zeros, ones, arange)
#. Use of blocks
#. Special Tensor class methods (e.g., random)

Converting Crystal Array's to Bottle Tensors
--------------------------------------------

In general, numerical data with a consistent shape stored in an Array
can be converted to a tensor using the ``from_array`` class method
of the ``Tensor`` class.  This method identifies the shape of strides
of the array, and if it is not jagged, returns a valid tensor.

Examples:

.. code-block:: crystal

    puts Tensor.from_array [2, 3, 1, 0]
    puts Tensor.from_array [[1, 2], [3, 4], [5, 6]]
    puts Tensor.from_array [[1, 2, 3], [4, 5]]

.. code-block:: crystal

    Tensor([2, 3, 1, 0])
    Tensor([[1, 2],
            [3, 4],
            [5, 6]])
    Unhandled exception: All subarrays must be the same length (Bottle::Internal::Exceptions::ShapeError)

Intrinsic Bottle Tensor Creation
--------------------------------

Bottle has many built-in routines for creating tensors.

``zeros(shape)`` will create a tensor filled with 0s with a specified shape.

.. code-block:: crystal

    puts B.zeros([2, 3])

.. code-block:: crystal

    Tensor([[0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]])

``ones(shape)`` will create an array with 1s, and is identical to ``zeros`` in every other way.

``arange()`` will create tensors with regularly incrementing values.  It will return a tensor
with a provided step (default is 1) that goes from a given start to end value.

.. code-block:: crystal

    puts B.arange(10)
    puts B.arange(2, 10, dtype: Float64)
    puts B.arange(2, 3, 0.1, dtype: Float64)

.. code-block:: crystal

    Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    Tensor([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])

``arange`` can produce tensors of unpredictable sizes when given a non-integer step.  To avoid this,
``linspace`` creates a fixed number of evenly spaced elements between two provided values.

.. code-block:: crystal

    puts B.linspace(1, 4, 6)

.. code-block:: crystal

    Tensor([1.0, 1.6, 2.2, 2.8, 3.4, 4.0])
