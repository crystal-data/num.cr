********
allclose
********

.. container:: entry-detail
   :name: allclose(a:Num::BaseArray(Bool),b:Num::BaseArray(Bool))-instance-method

   .. container:: signature

      def **allclose**\ (a : Num::BaseArray(Bool), b :
      Num::BaseArray(Bool))

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/testing/testing.cr#L36>`__]


.. container:: entry-detail
   :name: allclose(a:Num::BaseArray(U),b:Num::BaseArray(U),rtol=1e-5,atol=1e-8)forallU-instance-method

   .. container:: signature

      def **allclose**\ (a : Num::BaseArray(U), b : Num::BaseArray(U),
      rtol = 1e-5, atol = 1e-8) forall U

   .. container:: doc

      Asserts that two equally shaped ``Tensor``\ s are equal within a
      provided tolerance. Useful for floating point comparison where
      direct equality might not work

      ::

         t = Tensor.new([2, 2, 3]) { |i| i * 1.0 }
         tf = t + 0.00000000001
         allclose(t, tf) # => true

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/testing/testing.cr#L15>`__]
