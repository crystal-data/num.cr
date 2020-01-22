*******
Tensors
*******

Num.cr provides an n-dimensional Tensor, which stores numeric data of a
homogenous type.  These tensors can be :ref:`indexed <tensors.indexing>` using
a variety of indexing operations.

All Tensors contain a single data type, union types are disallowed in order
to maximize performance and clarity.  Tensors in particular only can store
boolean or numeric data, but BaseArray's can be subclassed in order to use
convenient n-dimensional storage for other types.

.. toctree::
   :maxdepth: 2

   tensors.indexing
