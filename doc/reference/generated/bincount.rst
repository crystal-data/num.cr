********
bincount
********

.. container:: entry-detail
   :name: bincount(x:Tensor(Int32),min_count=0)-instance-method

   .. container:: signature

      def **bincount**\ (x : Tensor(Int32), min_count = 0)

   .. container:: doc

      Count number of occurrences of each value in array of non-negative
      ints.

      The number of bins (of size 1) is one larger than the largest
      value in x. If minlength is specified, there will be at least this
      number of bins in the output array (though it will be longer if
      necessary, depending on the contents of x). Each bin gives the
      number of occurrences of its index value in x. If weights is
      specified the input array is weighted by it, i.e. if a value n is
      found at position i, out[n] += weight[i] instead of out[n] += 1.

      ::

         t = Tensor.random(0...10, [10])
         t           # => Tensor([7, 2, 2, 7, 0, 7, 6, 6, 0, 6])
         bincount(t) # => Tensor([2, 0, 2, 0, 0, 0, 3, 3, 0, 0])

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L314>`__]


.. container:: entry-detail
   :name: bincount(x:Tensor(Int32),weights:Tensor(U),min_count=0)forallU-instance-method

   .. container:: signature

      def **bincount**\ (x : Tensor(Int32), weights : Tensor(U),
      min_count = 0) forall U

   .. container:: doc

      Count number of occurrences of each value in array of non-negative
      ints.

      The number of bins (of size 1) is one larger than the largest
      value in x. If minlength is specified, there will be at least this
      number of bins in the output array (though it will be longer if
      necessary, depending on the contents of x). Each bin gives the
      number of occurrences of its index value in x. If weights is
      specified the input array is weighted by it, i.e. if a value n is
      found at position i, out[n] += weight[i] instead of out[n] += 1.

      ::

         t = Tensor.random(0...10, [10])
         t           # => Tensor([7, 2, 2, 7, 0, 7, 6, 6, 0, 6])
         bincount(t) # => Tensor([2, 0, 2, 0, 0, 0, 3, 3, 0, 0])

   .. container::

      [`View
      source <https://github.com/crystal-data/num.cr/blob/32a5d0701dd7cef3485867d2afd897900ca60901/src/tensor/creation.cr#L342>`__]
