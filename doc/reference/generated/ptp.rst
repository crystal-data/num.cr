***
ptp
***

.. container:: entry-detail
   :name: ptp(v:BaseArray)-instance-method

   .. container:: signature

      def **ptp**\ (v : BaseArray)

   .. container:: doc

      Computes the "peak to peak" of a BaseArray (max - min)

      ::

         v = BaseArray.new [1, 2, 3, 4]
         v.ptp # => 3

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L146>`__]


.. container:: entry-detail
   :name: ptp(a:BaseArray,axis:Int32)-instance-method

   .. container:: signature

      def **ptp**\ (a : BaseArray, axis : Int32)

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L150>`__]
