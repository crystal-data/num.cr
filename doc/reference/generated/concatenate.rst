***********
concatenate
***********

.. container:: entry-detail
   :name: concatenate(alist:Array(Num::BaseArray(U)))forallU-instance-method

   .. container:: signature

      def **concatenate**\ (alist : Array(Num::BaseArray(U))) forall U

   .. container:: doc

      Concatenates an array of one dimensional tensors.

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/assemble.cr#L87>`__]


.. container:: entry-detail
   :name: concatenate(alist:Array(Num::BaseArray(U)),axis:Int32)forallU-instance-method

   .. container:: signature

      def **concatenate**\ (alist : Array(Num::BaseArray(U)), axis :
      Int32) forall U

   .. container:: doc

      Join a sequence of arrays along an existing axis. The arrays must
      have the same shape, except in the dimension corresponding to axis
      (the first, by default).

      Parameters:

      -  alist : Array(BaseArray(U)) The arrays to concatenate
      -  axis : Int32 The axis along which to concatenate

      Return:

      -  BaseArray(U) - The concatenated arrays

      Example:

      ::

         a = Num.zeros([3, 3])
         b = Num.ones([3, 3])
         puts Num.concatenate([a, b], 1)

      Output

      ::

         Tensor([[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]])

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/assemble.cr#L55>`__]
