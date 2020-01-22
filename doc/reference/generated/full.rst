****
full
****

.. container:: entry-detail
   :name: full(shape:Array(Int32),x:Number,dtype:U.class=Float64)forallU-instance-method

   .. container:: signature

      def **full**\ (shape : Array(Int32), x : Number, dtype : U.class =
      Float64) forall U

   .. container:: doc

      Initializes a ``Tensor`` of the given ``size`` and ``dtype``,
      filled with the given value.

      ::

         f = full(5, 3, dtype: Int32)
         f # => Tensor[3, 3, 3, 3, 3]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L121>`__]
