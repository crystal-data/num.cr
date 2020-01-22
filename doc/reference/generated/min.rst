***
min
***

.. container:: entry-detail
   :name: min(a:Num::BaseArray(U))forallU-instance-method

   .. container:: signature

      def **min**\ (a : Num::BaseArray(U)) forall U

   .. container:: doc

      Computes the minimum value of a BaseArray

      ::

         v = BaseArray.new [1, 2, 3, 4]
         min(v) # => 1

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L118>`__]


.. container:: entry-detail
   :name: min(a:BaseArray,axis:Int32,keepdims=false)-instance-method

   .. container:: signature

      def **min**\ (a : BaseArray, axis : Int32, keepdims = false)

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L132>`__]
