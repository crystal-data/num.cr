**********
zeros_like
**********

.. container:: entry-detail
   :name: zeros_like(other:Tensor,dtype:U.class=Float64)forallU-instance-method

   .. container:: signature

      def **zeros_like**\ (other : Tensor, dtype : U.class = Float64)
      forall U

   .. container:: doc

      Initializes a ``Tensor`` filled with zeros, whose size is inferred
      from a given ``Tensor``

      ::

         t = Tensor.new [1, 2, 3]

         f = zeros_like(t, dtype: Int32)
         f # => Tensor[0, 0, 0]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L110>`__]
