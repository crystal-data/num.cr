*****
empty
*****

.. container:: entry-detail
   :name: empty(shape:Array(Int32),dtype:U.class=Float64)forallU-instance-method

   .. container:: signature

      def **empty**\ (shape : Array(Int32), dtype : U.class = Float64)
      forall U

   .. container:: doc

      Initializes a ``Tensor`` with an uninitialized slice of data.

      ::

         f = empty(5, dtype: Int32)
         f # => Tensor[0, 0, 0, 0, 0]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L19>`__]
