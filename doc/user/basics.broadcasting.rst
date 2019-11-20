************
Broadcasting
************

The term broadcasting describes how Bottle treats tensors with different shapes during arithmetic operations.
Subject to certain constraints, the smaller tensor is "broadcast" across the larger tensor so that they have compatible shapes.
It does this without making needless copies of data and usually leads to efficient algorithm implementations.
There are, however, cases where broadcasting is a bad idea because it leads to inefficient use of memory that slows computation.

Bottle operations are usually done on pairs of tensors on an element-by-element basis.
In the simplest case, the two tensors must have exactly the same shape, as in the following example:

.. code-block:: crystal

    a = Tensor.from_array [1.0, 2.0, 3.0]
    b = Tensor.from_array [2.0, 2.0, 2.0]

    puts a * b

.. code-block:: crystal

    Tensor([2.0, 4.0, 6.0])

Bottle's broadcasting rules relax this constraint when tensors meet certain requirements.  The simplest example
occurs when an array and a scalar value are combined in an operation.

.. code-block:: crystal

    a = Tensor.from_array [1.0, 2.0, 3.0]
    b = 2.0

    puts a * b

.. code-block:: crystal

    Tensor([2.0, 4.0, 6.0])

The result is equivalent to the previous example where b was a tensor. We can think
of the scalar b being stretched during the arithmetic operation into a tensor with the same shape as a.
The new elements in b are simply copies of the original scalar. The stretching analogy is only conceptual.
Bottle is smart enough to use the original scalar value without actually making copies,
so that broadcasting operations are as memory and computationally efficient as possible.

The code in the second example is more efficient than that in the first because broadcasting
moves less memory around during the multiplication (b is a scalar rather than a tensor).

General Broadcasting Rules
--------------------------

When operating on two arrays, Bottle compares their shapes element-wise. It starts with the trailing
dimensions, and works its way forward. Two dimensions are compatible when

#. they are equal, or
#. one of them is 1

If these conditions are not met, a ShapeError is thrown, indicating that the tensors have incompatible shapes.
The size of the resulting array is the maximum size along each dimension of the input tensors.

Tensors do not need to have the same number of dimensions. For example, if you have a 256x256x3 tensor of RGB values,
and you want to scale each color in the image by a different value, you can multiply the image by a
one-dimensional tensor with 3 values. Lining up the sizes of the trailing axes of these tensors
according to the broadcast rules, shows that they are compatible:

.. code-block:: crystal

    Image  (3d tensor): 256 x 256 x 3
    Scale  (1d tensor):             3
    Result (3d tensor): 256 x 256 x 3

When either of the dimensions compared is one, the other is used. In other words,
dimensions with size 1 are stretched or "copied" to match the other.

In the following example, both the A and B tensors have axes with length one
that are expanded to a larger size during the broadcast operation:

.. code-block:: crystal

    A      (4d tensor):  8 x 1 x 6 x 1
    B      (3d tensor):      7 x 1 x 5
    Result (4d tensor):  8 x 7 x 6 x 5

Here are more examples:

.. code-block:: crystal

    A      (2d tensor):  5 x 4
    B      (1d tensor):      1
    Result (2d tensor):  5 x 4

    A      (2d tensor):  5 x 4
    B      (1d tensor):      4
    Result (2d tensor):  5 x 4

    A      (3d tensor):  15 x 3 x 5
    B      (3d tensor):  15 x 1 x 5
    Result (3d tensor):  15 x 3 x 5

    A      (3d tensor):  15 x 3 x 5
    B      (2d tensor):       3 x 5
    Result (3d tensor):  15 x 3 x 5

    A      (3d tensor):  15 x 3 x 5
    B      (2d tensor):       3 x 1
    Result (3d tensor):  15 x 3 x 5

Broadcasting provides a convenient way of taking the outer product (or any other outer operation)
of two tensors. The following example shows an outer addition operation of two 1-d tensors:

.. code-block:: crystal

    a = B.arange(4) * 10
    b = B.arange(3)

    puts a.bc?(1) + b

.. code-block:: crystal

    Tensor([[ 0,  1,  2],
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32]])
