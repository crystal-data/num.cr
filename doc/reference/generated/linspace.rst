********
linspace
********

.. container:: entry-detail
   :name: linspace(start:Number,stop:Number,num=50,endpoint=true)-instance-method

   .. container:: signature

      def **linspace**\ (start : Number, stop : Number, num = 50,
      endpoint = true)

   .. container:: doc

      Return evenly spaced numbers over a specified interval. Returns
      ``num`` evenly spaced samples, calculated over the interval
      [``start``, ``stop``]. The endpoint of the interval can optionally
      be excluded.

      ::

         B.linspace(0, 1, 5) # => Tensor[0.0, 0.25, 0.5, 0.75, 1.0]

         B.linspace(0, 1, 5, endpoint: false) # => Tensor[0.0, 0.2, 0.4, 0.6, 0.8]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L190>`__]
