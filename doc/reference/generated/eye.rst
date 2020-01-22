***
eye
***

.. container:: entry-detail
   :name: eye(m:Int32,n:Int32?=nil,k:Int32=0,dtype:U.class=Float64)forallU-instance-method

   .. container:: signature

      def **eye**\ (m : Int32, n : Int32? = nil, k : Int32 = 0, dtype :
      U.class = Float64) forall U

   .. container:: doc

      Return a ``Matrix`` with ones on the diagonal and zeros elsewhere.

      ::

         m = eye(3, dtype: Int32)

         m # => [[1, 0, 0], [0, 1, 0], 0, 0, 1]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L45>`__]
