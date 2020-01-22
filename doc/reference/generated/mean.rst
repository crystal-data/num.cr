****
mean
****

.. container:: entry-detail
   :name: mean(a:BaseArray)-instance-method

   .. container:: signature

      def **mean**\ (a : BaseArray)

   .. container:: doc

      Computes the average of all BaseArray values

      ::

         v = BaseArray.new [1, 2, 3, 4]
         mean(v) # => 2.5

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L58>`__]


.. container:: entry-detail
   :name: mean(a:Num::BaseArray(U),axis:Int32,keepdims=false)forallU-instance-method

   .. container:: signature

      def **mean**\ (a : Num::BaseArray(U), axis : Int32, keepdims =
      false) forall U

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/reductions.cr#L62>`__]
