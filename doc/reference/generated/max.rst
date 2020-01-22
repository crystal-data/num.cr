***
max
***

.. container:: entry-detail
   :name: max(a:BaseArray,axis:Int32,keepdims=false)-instance-method

   .. container:: signature

      def **max**\ (a : BaseArray, axis : Int32, keepdims = false)

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L104>`__]


.. container:: entry-detail
   :name: max(a:Num::BaseArray(U))forallU-instance-method

   .. container:: signature

      def **max**\ (a : Num::BaseArray(U)) forall U

   .. container:: doc

      Computes the maximum value of a BaseArray

      ::

         v = BaseArray.new [1, 2, 3, 4]
         max(v) # => 4

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L90>`__]
