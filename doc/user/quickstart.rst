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

Num.cr's main object is the homogeneous multidimensional tensor. It is a
table of elements (usually numbers), all of the same type, indexed by
integers. In Num.cr dimensions are called *axes*.

For example, the coordinates of a point in 3D space ``[1, 2, 1]`` has
one axis. That axis has 3 elements in it, so we say it has a length
of 3. In the example pictured below, the tensor has 2 axes. The first
axis has a length of 2, the second axis has a length of 3.

::

    Tensor([[ 1.0, 0.0, 0.0],
            [ 0.0, 1.0, 2.0]])


Num.cr's tensor class is called ``Tensor``. The more important attributes of
a ``tensor`` object are:

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

    t = Num.arange(15).reshape([3, 5])

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
    Tensor(Int32)
    Tensor([6, 7, 8])
    Tensor(Int32)

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
Num.cr offers many routines to create tensors with initial placeholder data.  These
minimize the number of tensors that need to grow to fit data, which is an
expensive operation.

The routine ``zeros`` creates a tensor full of zeros.  The routine ``ones`` creates
a tensor full of ones, and the function ``empty`` creates an array with an empty
allocated data buffer.

.. code-block:: crystal

    puts Num.zeros([3, 4])
    puts Num.ones([2, 3, 4], dtype: UInt8)
    puts Num.empty([2, 3])

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

To create sequences of numbers, Num.cr provides functions similar to ranges that return
tensors instead of iterators.

.. code-block:: crystal

    puts Num.arange(10, 30, 5)
    puts Num.arange(0, 2, 0.3, dtype: Float64)

.. code-block:: crystal

    Tensor([10, 15, 20, 25])
    Tensor([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])

When ``arange`` is used with floating point arguments, it is generally not possible to predict
the number of elements created, due to floating point precision issues.  It is usually better
to use the ``linspace`` routine that receives as an argument the number of desired elements, instead of a
step.

.. code-block:: crystal

    puts Num.linspace(0, 2, 9)

.. code-block:: crystal

    Tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

Basic Operations
----------------

Arithmetic operations on tensors apply *elementwise*.  A new tensor is created and filled
with the result.

.. code-block:: crystal

    a = Tensor.from_array [20, 30, 40, 50]
    b = Num.arange(4)
    puts b
    c = a - b
    puts c
    puts b ** 2
    puts Num.sin(a) * 10
    puts a < 35

.. code-block:: crystal

    Tensor([0, 1, 2, 3])
    Tensor([20, 29, 38, 47])
    Tensor([0, 1, 4, 9])
    Tensor([9.129, -9.88, 7.451, -2.624])
    Tensor([ true,  true, false, false])

Many statistical operations, such as the sum of a tensor, or the minimum/maximum are implemented
directly as methods on the tensor class.

.. code-block:: crystal

    a = Tensor.random(0.0...1.0, [2, 3])
    puts a
    puts a.sum
    puts a.min
    puts a.max

.. code-block:: crystal

    Tensor([[0.064, 0.533, 0.395],
            [0.017, 0.025, 0.816]])
    1.8505205175980595
    0.017296349857875204
    0.816088601545241

By default, these operations treat the tensor as though it was a flattened version
of itself, returning a reduction on the entire tensor.  However, by specifying
and ``axis`` parameter, you can apply an operation along a specified access of a tensor.

.. code-block:: crystal

    b = Num.arange(12).reshape([3, 4])
    puts b

    puts b.sum(axis: 0)
    puts b.min(axis: 1)
    puts b.cumsum(axis: 1)

.. code-block:: crystal

    Tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    Tensor([12, 15, 18, 21])
    Tensor([0, 4, 8])
    Tensor([[ 0,  1,  3,  6],
            [ 4,  9, 15, 22],
            [ 8, 17, 27, 38]])


Universal Functions
-------------------

Num.cr provides familiar mathematical functions such as sin, cos, and exp.  These functions
operate elementwise on tensors, producing tensors as output.

.. code-block:: crystal

    t = Num.arange(3)
    puts t

    puts Num.exp(t)
    puts Num.sqrt(t)

    c = Tensor.from_array [2.0, -1.0, 4.0]

    puts Num.add(t, c)

.. code-block:: crystal

    Tensor([0, 1, 2])
    Tensor([  1.0, 2.718, 7.389])
    Tensor([  0.0,   1.0, 1.414])
    Tensor([2.0, 0.0, 6.0])

Indexing, Slicing and Iterating
-------------------------------

**One-dimensional** tensors can be indexed, sliced and iterated over, very similar to
Crystal arrays.

.. code-block:: crystal

    a = Num.arange(10) ** 3
    puts a

    puts a[[2]]
    puts a[2...5]

.. code-block:: crystal

    Tensor([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
    8
    Tensor([ 8, 27, 64])

**N-Dimensional** tensors can have a single index operation per axis. These indices are provided
as `*args`.

.. code-block:: crystal

    a = Tensor.new(5, 4) do |i, j|
      10 * i + j
    end
    puts a

    puts a[2, 3].value
    puts a[...5, 1]
    puts a[..., 1]
    puts a[1...3, ...]

.. code-block:: crystal

    Tensor([[ 0,  1,  2,  3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43]])
    23
    Tensor([ 1, 11, 21, 31, 41])
    Tensor([ 1, 11, 21, 31, 41])
    Tensor([[10, 11, 12, 13],
            [20, 21, 22, 23]])


Shape Manipulation
==================

Changing the shape of a tensor
------------------------------

Tensors have shapes defined by the number of elements along each axis.

.. code-block:: crystal

    a = Tensor.random(0...10, [3, 4])
    puts a
    puts a.shape

.. code-block:: crystal

    Tensor([[8, 4, 8, 5],
            [7, 5, 9, 5],
            [3, 7, 5, 5]])
    [3, 4]

The shape of a atensor can be changed with many routines.  Many methods return
a view of the original data, but do not change the origin tensor.

.. code-block:: crystal

    puts a.ravel
    puts a.reshape([6, 2])
    puts a.transpose
    puts a.transpose.shape
    puts a.shape

.. code-block:: crystal

    Tensor([8, 4, 8, 5, 7, 5, 9, 5, 3, 7, 5, 5])
    Tensor([[8, 4],
            [8, 5],
            [7, 5],
            [9, 5],
            [3, 7],
            [5, 5]])
    Tensor([[8, 7, 3],
            [4, 5, 7],
            [8, 9, 5],
            [5, 5, 5]])
    [4, 3]
    [3, 4]

If a dimension is provided as -1 in an operation that reshapes the tensor, the other dimensions
are calculated automatically.  Only a single dimension can be dynamically calculated.

.. code-block:: crystal

    puts a.reshape(3, 2, -1)

.. code-block:: crystal

    Tensor([[[8, 4],
             [8, 5]],

            [[7, 5],
             [9, 5]],

            [[3, 7],
             [5, 5]]])


Stacking together different tensors
-----------------------------------

Many tensors can be stacked together along an axis.  Shapes must be the same on
the off-axis dimensions of the tensors.

.. code-block:: crystal

    a = Tensor.random(0...10, [2, 2])
    b = Tensor.random(0...10, [2, 2])

    puts a
    puts b

    puts Num.vstack([a, b])
    puts Num.hstack([a, b])
    puts Num.column_stack([a, b])

.. code-block:: crystal

    Tensor([[7, 7],
            [1, 3]])
    Tensor([[3, 9],
            [7, 0]])
    Tensor([[7, 7],
            [1, 3],
            [3, 9],
            [7, 0]])
    Tensor([[7, 7, 3, 9],
            [1, 3, 7, 0]])
    Tensor([[7, 7, 3, 9],
            [1, 3, 7, 0]])

Copies and Views
================

When operating and manipulating tensors, data is sometimes copied into a new tensor, and
sometimes a tensor shares memory with another tensor.  This can lead to confusing
behavior if a user is not aware of this fact.

No copy at all
--------------

Simple assignments make no copy of tensors or their data

.. code-block:: crystal

    a = Num.arange(12).reshape([3, 4])
    b = a  # no copy of the tensors data is made


View or Shallow Copy
--------------------

Different tensors can share the same data, however some tensors will point to subsets
of another tensors data, and therefore the objects will not be the same.

.. code-block:: crystal

    c = a.dup_view()

    puts a.buffer == c.buffer
    puts c.flags.own_data?

    c = c.reshape([2, 6])
    c[0, 4] = 12345
    puts a

.. code-block:: crystal

    true
    false
    Tensor([[    0,     1,     2,     3],
            [12345,     5,     6,     7],
            [    8,     9,    10,    11]])

Slicing tensors returns a view

.. code-block:: crystal

    s = a[..., 1...3]
    s[...] = 10

    puts a

.. code-block:: crystal

    Tensor([[    0,    10,    10,     3],
            [12345,    10,    10,     7],
            [    8,    10,    10,    11]])

Deep copies
-----------

The ``dup`` method makes a copy of a tensor and its data

.. code-block:: crystal

    d = a.dup
    puts d.buffer == a.buffer

    d[0, 0] = 9999
    puts a

.. code-block:: crystal

    false
    Tensor([[    0,    10,    10,     3],
            [12345,    10,    10,     7],
            [    8,    10,    10,    11]])
