===================
Quickstart tutorial
===================

Prerequisites
=============

Before reading this tutorial you should know Crystal Lang. If you
would like to refresh your memory, take a look at the `Crystal
introduction <https://crystal-lang.org/reference/>`__.

The Basics
==========

Bottle's main object is the homogeneous multidimensional tensor. It is a
table of elements (usually numbers), all of the same type, indexed by
integers. In Bottle dimensions are called *axes*.

For example, the coordinates of a point in 3D space ``[1, 2, 1]`` has
one axis. That axis has 3 elements in it, so we say it has a length
of 3. In the example pictured below, the tensor has 2 axes. The first
axis has a length of 2, the second axis has a length of 3.

::

    Tensor([[ 1.0, 0.0, 0.0],
            [ 0.0, 1.0, 2.0]])


Bottle's tensor class is called ``Tensor``. The more important attributes of
an ``ndarray`` object are:

Tensor.ndims
    the number of axes (dimensions) of the tensor.
Tensor.shape
    the dimensions of the tensor. This is an Array of integers indicating
    the size of the tensor in each dimension. For a matrix with *n* rows
    and *m* columns, ``shape`` will be ``[n, m]``. The length of the
    ``shape`` Array is therefore the number of axes, ``ndims``.
Tensor.size
    the total number of elements of the tensor. This is equal to the
    product of the elements of ``shape``.
Tensor.buffer
    the buffer containing the actual elements of the array. Normally, we
    won't need to use this attribute because we will access the elements
    in an array using indexing facilities.

An example
----------

.. code-block:: crystal

    t = B.arange(15).reshape([3, 5])

    puts t
    puts t.shape
    puts t.ndims
    puts t.size
    puts typeof(t)

    b = Tensor.from_array [6, 7, 8]

    puts b
    puts typeof(b)

.. code-block:: crystal

    Tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]])
    [3, 5]
    2
    15
    Bottle::Tensor(Int32)
    Tensor([6, 7, 8])
    Bottle::Tensor(Int32)

Tensor Creation
---------------

There are many ways to create Tensors.

Tensors can be created from standard Crystal arrays, using the class method:
``from_array``.

.. code-block:: crystal

    t = Tensor.from_array [[1, 2, 3], [4, 5, 6]]
    puts t

.. code-block:: crystal

    Tensor([[1, 2, 3],
            [4, 5, 6]])

Tensors can also be created from blocks, allowing convenient construction of
complex shapes.

.. code-block:: crystal

    t = Tensor.new([3, 2, 2]) { |i| i**2 }
    puts t

.. code-block:: crystal

    Tensor([[[  0,   1],
             [  4,   9]],

            [[ 16,  25],
             [ 36,  49]],

            [[ 64,  81],
             [100, 121]]])

Often, the elements of an tensor are originally unknown, but its size of known.  Hence,
Bottle offers many routines to create tensors with initial placeholder data.  These
minimize the number of tensors that need to grow to fit data, which is an
expensive operation.

The routine ``zeros`` creates a tensor full of zeros.  The routine ``ones`` creates
a tensor full of ones, and the function ``empty`` creates an array with an empty
allocated data buffer.

.. code-block:: crystal

    puts B.zeros([3, 4])
    puts B.ones([2, 3, 4], dtype: UInt8)
    puts B.empty([2, 3])

.. code-block:: crystal

    Tensor([[0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]])
    Tensor([[[1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1]],

            [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1]]])
    Tensor([[0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]])
