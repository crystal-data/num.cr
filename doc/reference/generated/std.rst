***
std
***

.. container:: entry-detail
   :name: std(a:BaseArray)-instance-method

   .. container:: signature

      def **std**\ (a : BaseArray)

   .. container:: doc

      Computes the standard deviation of a BaseArray

      ::

         v = BaseArray.new [1, 2, 3, 4]
         std(v) # => 1.118

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L72>`__]


.. container:: entry-detail
   :name: std(a:BaseArray,axis:Int32,keepdims=false)-instance-method

   .. container:: signature

      def **std**\ (a : BaseArray, axis : Int32, keepdims = false)

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L78>`__]
