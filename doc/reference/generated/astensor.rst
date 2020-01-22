********
astensor
********

.. container:: entry-detail
   :name: astensor(a:Array)-instance-method

   .. container:: signature

      def **astensor**\ (a : Array)

   .. container:: doc

      Converts input data, in any form that can be converted to a
      tensor, into a tensor.

      This includes arrays, nested arrays, scalars, and tensors. Data
      will not be copied unless necessary. Base classes will not be
      maintained, all inputs will be coerced to Tensors or raise.

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/converters.cr#L30>`__]


.. container:: entry-detail
   :name: astensor(a:Number)-instance-method

   .. container:: signature

      def **astensor**\ (a : Number)

   .. container:: doc

      Converts input data, in any form that can be converted to a
      tensor, into a tensor.

      This includes arrays, nested arrays, scalars, and tensors. Data
      will not be copied unless necessary. Base classes will not be
      maintained, all inputs will be coerced to Tensors or raise.

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/converters.cr#L40>`__]


.. container:: entry-detail
   :name: astensor(a:Tensor)-instance-method

   .. container:: signature

      def **astensor**\ (a : Tensor)

   .. container:: doc

      Converts input data, in any form that can be converted to a
      tensor, into a tensor.

      This includes arrays, nested arrays, scalars, and tensors. Data
      will not be copied unless necessary. Base classes will not be
      maintained, all inputs will be coerced to Tensors or raise.

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/converters.cr#L10>`__]


.. container:: entry-detail
   :name: astensor(a:Num::BaseArray(U))forallU-instance-method

   .. container:: signature

      def **astensor**\ (a : Num::BaseArray(U)) forall U

   .. container:: doc

      Converts input data, in any form that can be converted to a
      tensor, into a tensor.

      This includes arrays, nested arrays, scalars, and tensors. Data
      will not be copied unless necessary. Base classes will not be
      maintained, all inputs will be coerced to Tensors or raise.

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/core/converters.cr#L20>`__]
