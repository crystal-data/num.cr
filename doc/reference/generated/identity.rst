********
identity
********

.. container:: entry-detail
   :name: identity(n:Int32,dtype:U.class=Float64)forallU-instance-method

   .. container:: signature

      def **identity**\ (n : Int32, dtype : U.class = Float64) forall U

   .. container:: doc

      Returns the identify matrix with dimensions *m* by *m*

      ::

         m = identity(3)

         m # => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L60>`__]
