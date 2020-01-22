**********
empty_like
**********

.. container:: entry-detail
   :name: empty_like(other:Tensor,dtype:U.class=Float64)forallU-instance-method

   .. container:: signature

      def **empty_like**\ (other : Tensor, dtype : U.class = Float64)
      forall U

   .. container:: doc

      Initializes a ``Tensor`` with an uninitialized slice of data that
      is the same size as a given ``Tensor``.

      ::

         t = Tensor.new [1, 2, 3]

         f = empty_like(t, dtype: Int32)
         f # => Tensor[0, 0, 0]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L33>`__]
