.. _tensor.ndtensor:

******************************************
The N-dimensional tensor
******************************************

An :class:`tensor` is a (usually fixed-size) multidimensional
container of items of the same type and size. The number of dimensions
and items in an array is defined by its :attr:`shape`,
which is a :class:`array` of *N* non-negative integers that specify the
sizes of each dimension. The type of items in the array is specified by
a separate :ref:`data-type object`, one of which
is associated with each Tensor.

As with other container objects in Crystal, the contents of an
:class:`tensor` can be accessed and modified by :ref:`indexing or
slicing <tensor.indexing>` the tensor (using, for example, *N* integers),
and via the methods and attributes of the :class:`tensor`.

.. index:: view, base

Different :class:`tensors <tensor>` can share the same data, so that
changes made in one :class:`tensor` may be visible in another. That
is, a tensor can be a *"view"* to another tensor, and the data it
is referring to is taken care of by the *"base"* pointer.


.. admonition:: Example

   A 2-dimensional tensor of size 2 x 3, composed of 4-byte integer
   elements:

.. code-block:: crystal

   x = Tensor.from_array [[1, 2, 3], [4, 5, 6]]
   puts typeof(x)
   puts x.shape
   puts x.dtype

.. code-block:: crystal

   Tensor(Int32)
   [2, 3]
   Int32


The array can be indexed using array-like indexing syntax:

.. code-block:: crystal

   # The element of x in the *second* row, *third* column, namely, 6.
   x[1, 2]


For example :ref:`slicing <tensor.indexing>` can produce views of
the tensor:

.. code-block:: crystal

   y = x[...,1]
   puts y
   y[0] = 9
   puts y
   puts x

.. code-block:: crystal

   Tensor([2, 5])
   Tensor([9, 5])
   Tensor([1, 9, 3],
          [4, 5, 6])


Constructing Tensors
===================

New tensors can be constructed using the routines detailed in
:ref:`routines.tensor-creation`, and also by using the low-level
:class:`tensor` constructor:

.. autosummary::
   :toctree: generated/

   tensor

.. _tensor.ndtensor.indexing:


Indexing arrays
===============

Arrays can be indexed using an extended Crystal slicing syntax,
``tensor[selection]``.

.. seealso:: :ref:`Tensor Indexing <tensor.indexing>`.

.. _memory-layout:

Internal memory layout of an Tensor
====================================

An instance of class :class:`tensor` consists of a contiguous
one-dimensional segment of computer memory (owned by the tensor, or by
some other tensor), combined with an indexing scheme that maps *N*
integers into the location of an item in the block.  The ranges in
which the indices can vary is specified by the :obj:`shape
<tensor.shape>` of the array. How many bytes each item takes and how
the bytes are interpreted is defined by the :ref:`dtype` associated with the tensor.

.. index:: C-order, Fortran-order, row-major, column-major, stride,
  offset

A segment of memory is inherently 1-dimensional, and there are many
different schemes for arranging the items of an *N*-dimensional tensor
in a 1-dimensional block. Bottle is flexible, and :class:`tensor`
objects can accommodate any *strided indexing scheme*. In a strided
scheme, the N-dimensional index :math:`(n_0, n_1, ..., n_{N-1})`
corresponds to the offset (in bytes):

.. math:: n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k

from the beginning of the memory block associated with the
tensor. Here, :math:`s_k` are integers which specify the :obj:`strides
<tensor.strides>` of the tensor. The :term:`column-major` order (used,
for example, in the Fortran language and in *Matlab*) and
:term:`row-major` order (used in C) schemes are just specific kinds of
strided scheme, and correspond to memory that can be *addressed* by the strides:

.. math::

   s_k^{\mathrm{column}} = \mathrm{itemsize} \prod_{j=0}^{k-1} d_j ,
   \quad  s_k^{\mathrm{row}} = \mathrm{itemsize} \prod_{j=k+1}^{N-1} d_j .

.. index:: single-segment, contiguous, non-contiguous

where :math:`d_j` `= shape[j]`.

Both the C and Fortran orders are :term:`contiguous`, *i.e.,*
single-segment, memory layouts, in which every part of the
memory block can be accessed by some combination of the indices.

While a C-style and Fortran-style contiguous tensor, which has the corresponding
flags set, can be addressed with the above strides, the actual strides may be
different. This can happen in two cases:

    1. If ``shape[k] == 1`` then for any legal index ``index[k] == 0``.
       This means that in the formula for the offset :math:`n_k = 0` and thus
       :math:`s_k n_k = 0` and the value of :math:`s_k` `= self.strides[k]` is
       arbitrary.
    2. If a tensor has no elements (``size == 0``) there is no legal
       index and the strides are never used. Any tensor with no elements may be
       considered C-style and Fortran-style contiguous.

Point 1. means that even a high dimensional tensor could be C-style and Fortran-style
contiguous at the same time.

.. warning::

    It does *not* generally hold that ``strides[-1] == sizeof(dtype)``
    for C-style contiguous tensors or ``strides[0] == sizeof(dtype)`` for
    Fortran-style contiguous tensors is true.

Data in new :class:`tensors <tensor>` is in the :term:`row-major`
(C) order, unless otherwise specified, but, for example, :ref:`basic
array slicing <tensor.indexing>` often produces :term:`views <view>`
in a different scheme.

.. seealso: :ref:`Indexing <tensor.ndtensor.indexing>`_

.. note::

   Several algorithms in Bottle work on arbitrarily strided arrays.
   However, some algorithms require single-segment arrays. When an
   irregularly strided array is passed in to such algorithms, a copy
   is automatically made.

.. _tensor.ndtensor.attributes:
