********
logspace
********

.. container:: entry-detail
   :name: logspace(start,stop,num=50,endpoint=true,base=10.0)-instance-method

   .. container:: signature

      def **logspace**\ (start, stop, num = 50, endpoint = true, base =
      10.0)

   .. container:: doc

      Return numbers spaced evenly on a log scale. In linear space, the
      sequence starts at ````\ base \*\* start\ ```` (``base`` to the
      power of ``start``) and ends with ````\ base \*\* stop\ ```` (see
      ``endpoint`` below).

      ::

         B.logspace(2.0, 3.0, num = 4) # => Tensor[100.0, 215.44346900318845, 464.15888336127773, 1000.0]

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L223>`__]
