****
ones
****

.. container:: entry-detail
   :name: ones(shape:Array(Int32),dtype:U.class=Float64)forallU-instance-method

   .. container:: signature

      def **ones**\ (shape : Array(Int32), dtype : U.class = Float64)
      forall U

   .. container:: doc

      Initializes a ``Tensor`` of the given ``size`` and ``dtype``,
      filled with ones.

      ::

         f = ones(5, dtype: Int32)
         f # => Tensor[1, 1, 1, 1, 1]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L73>`__]
