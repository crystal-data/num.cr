********
Indexing
********

Tensor indexing referes to any use of the square brackets ([]) to access tensor values.
There are many options to indexing, which can lead to some potential for confusion.
This section is an overview of the various options and issues related to indexing.

Assignment vs referencing
-------------------------

Most of the following examples show the use of indexing when referencing data in a tensor. The examples
work just as well when assigning to tensors. See the section at the end for specific examples and
explanations on how assignments work.

Single Element Indexing
-----------------------

Single element indexing always is handled by passing an **array** of indices.  Unlike methods
that return a slice of a tensor, the methods that return or assign to a scalar are special
to avoid return type ambiguity.

.. code-block:: crystal

    x = B.arange(10)
    puts x[[2]]
    puts x[[-2]]

.. code-block:: crystal

    2
    8

This behavior similarly translates to multi-dimensional tensors.  When accessing a **scalar**
value, an array must be passed as the single argument.

.. code-block:: crystal

    x = x.reshape([2, 5])
    puts x[[1, 3]]
    puts x[[1, -1]]

.. code-block:: crystal

    8
    9

If a multi-dimensional array is index using ``*args``, the number of indexers must
not resolve to a scalar, and the number of passed indexers must be less than or
equal to the dimensions of the tensor.

.. code-block:: crystal

    puts x[0]

.. code-block:: crystal

    Tensor([0, 1, 2, 3, 4])

That is, each index specified selects the tensor corresponding to the rest of the dimensions selected.
In the above example, choosing 0 means that the remaining dimension of length 5 is being left
unspecified, and that what is returned is a tensor of that dimensionality and size.

A common mistake to make is the use of "chained indexing".  For example:

.. code-block:: crystal

    puts x[0][[2]]  # 2

However, this allocates an intermediate tensor, which is highly inefficient.  Instead, ``x[[0, 1]]``
should be used.

Other Indexing Options
----------------------

It is possible to slice and stride arrays of the same number of dimensions but of different sizes than
the original.

.. code-block:: crystal

    x = B.arange(10)
    puts x[2...5]
    puts x[...-7]

    y = B.arange(35).reshape([5, 7])
    puts y[1..., 2...4]

.. code-block:: crystal

    Tensor([2, 3, 4])
    Tensor([0, 1, 2])
    Tensor([[ 9, 10],
            [16, 17],
            [23, 24],
            [30, 31]])

Note that slices of tensors do not copy the internal tensor data but only produce new views of the original data.

Assigning values to indexed tensors
-----------------------------------

Tensors support assignment to slices of a tensor, using any of the above index
operations.  The value being assigned to the index array must be the same shape
(or broadcastable to the same shape) as the shape produced by the index operation.

For example, a scalar can be assigned to a slice.

.. code-block:: crystal

    x = B.arange(10)
    x[2...7] = 1

Another tensor of the right size can also be assigned to a view.

    x[2...7] = B.arange(5)


Assignments might result in the casting of the provided values.  Tensors
will always maintain their data type when have values assigned.

.. code-block:: crystal

    x = B.arange(10, dtype: Int32)
    x[[3]] = 999.6
    puts x

.. code-block:: crystal

    Tensor([  0,   1,   2, 999,   4,   5,   6,   7,   8,   9])

If a tensor is being assigned to a view of a different shape, the operation
will fail *unless* the tensor is able to be broadcasted to the proper shape.

.. code-block:: crystal

    t = B.arange(9).reshape([3, 3])
    puts t

    t[...] = B.arange(3)
    puts t

.. code-block:: crystal

    Tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    Tensor([[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]])
