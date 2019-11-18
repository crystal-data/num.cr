.. _tensor:

*************
Tensor Objects
*************

Bottle implements an N-Dimensional container called a Tensor.  This container
owns a single contiguous block of memory, and accessed elements based on
its `shape` and `strides` attributes.  These tensors can be sliced and indexed
to create "views" of tensors that do not own their own data.  These "views"
track where they get their data from through their base attribute.



.. toctree::
   :maxdepth: 2

   tensor.ndtensor
   tensor.dtypes
   tensor.indexing
