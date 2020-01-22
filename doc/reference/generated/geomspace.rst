*********
geomspace
*********

.. container:: entry-detail
   :name: geomspace(start,stop,num=50,endpoint=true)-instance-method

   .. container:: signature

      def **geomspace**\ (start, stop, num = 50, endpoint = true)

   .. container:: doc

      Return numbers spaced evenly on a log scale (a geometric
      progression). This is similar to ``#logspace``, but with endpoints
      specified directly. Each output sample is a constant multiple of
      the previous.

      ::

         geomspace(1, 1000, 4) # => Tensor[1.0, 10.0, 100.0, 1000.0]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L235>`__]
