*********
full_like
*********

.. container:: entry-detail
   :name: full_like(other:NDTensor,x:Number,dtype:U.class=Float64)forallU-instance-method

   .. container:: signature

      def **full_like**\ (other : NDTensor, x : Number, dtype : U.class
      = Float64) forall U

   .. container:: doc

      Initializes a ``Tensor`` filled with the provided value, whose
      size is inferred from a given ``Tensor``

      ::

         t = Tensor.new [1, 2, 3]

         f = full_like(t, -1, dtype: Int32)
         f # => Tensor[-1, -1, -1]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L134>`__]
