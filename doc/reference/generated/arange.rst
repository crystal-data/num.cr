******
arange
******

.. container:: entry-detail
   :name: arange(start:Int32,stop:Int32,step:Number=1,dtype:U.class=Int32)forallU-instance-method

   .. container:: signature

      def **arange**\ (start : Int32, stop : Int32, step : Number = 1,
      dtype : U.class = Int32) forall U

   .. container:: doc

      Return evenly spaced values within a given interval.

      Values are generated within the half-open interval [start, stop)
      (in other words, the interval including start but excluding stop).

      ::

         B.arange(1, 5) # => Tensor[1, 2, 3, 4]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L159>`__]


.. container:: entry-detail
   :name: arange(stop:Int32,step:Int32=1,dtype:U.class=Int32)forallU-instance-method

   .. container:: signature

      def **arange**\ (stop : Int32, step : Int32 = 1, dtype : U.class =
      Int32) forall U

   .. container:: doc

      Return evenly spaced values within a given interval.

      Values are generated within the half-open interval [start, stop)
      (in other words, the interval including start but excluding stop).

      ::

         B.arange(5) # => Tensor[0, 1, 2, 3, 4]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L176>`__]
