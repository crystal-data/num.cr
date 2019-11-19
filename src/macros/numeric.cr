require "../arrayops/math"
require "../arrayops/statistics"

module Bottle::Internal::Macros
  extend self

  macro has_numeric_ops(klass)
    # Elementwise addition of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def +(other)
      BMath.add(self, other)
    end

    # Elementwise subtraction of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def -(other)
      BMath.subtract(self, other)
    end

    # Elementwise multiplication of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def *(other)
      BMath.multiply(self, other)
    end

    def **(other)
      BMath.power(self, other)
    end

    # Elementwise division of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def /(other)
      BMath.divide(self, other)
    end

    # Elementwise modulus of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def %(other)
      BMath.modulo(self, other)
    end
  end

  macro has_bitwise_ops(klass)
    # Elementwise bitwise and of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def &(other)
      Binary.bitwise_and(self, other)
    end

    # Elementwise bitwise or of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def |(other)
      Binary.bitwise_or(self, other)
    end

    # Elementwise bitwise xor of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def ^(other)
      Binary.bitwise_xor(self, other)
    end
  end

  macro has_shift_ops(klass)
    # Elementwise left shift of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def <<(other)
      Binary.left_shift(self, other)
    end

    # Elementwise right shift of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def >>(other)
      Binary.right_shift(self, other)
    end
  end

  macro has_comparison_ops(klass)
    # Elementwise greater than of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def >(other)
      BMath.greater(self, other)
    end

    # Elementwise greater equal than of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def >=(other)
      BMath.greater_equal(self, other)
    end

    # Elementwise less than of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def <(other)
      BMath.less(self, other)
    end

    # Elementwise less equals of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def <=(other)
      BMath.less_equal(self, other)
    end

    # Elementwise equals of a {{klass}}} to another equally
    # sized {{klass}}} or scalar
    def ==(other)
      BMath.equal(self, other)
    end
  end

  macro has_reduction_ops(klass)
    def cumsum(axis : Int32)
      BMath.add.accumulate(self, axis)
    end
  end

  macro has_statistical_ops(klass)
    # Computes the total sum of a Tensor
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # sum(v) # => 10
    # ```
    def sum
      Statistics.sum(self)
    end

    # Computes the total sum of a Tensor
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # sum(v) # => 10
    # ```
    def sum(axis : Int32)
      Statistics.sum(self, axis)
    end

    # Computes the average of all Tensor values
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # mean(v) # => 2.5
    # ```
    def mean
      Statistics.mean(self)
    end

    # Computes the average of all Tensor values
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # mean(v) # => 2.5
    # ```
    def mean(axis : Int32)
      Statistics.mean(self, axis)
    end

    # Computes the standard deviation of a Tensor
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # std(v) # => 1.118
    # ```
    def std
      Statistics.std(self)
    end

    # Computes the median value of a Tensor
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # median(v) # => 2.5
    # ```
    def median
      Statistics.median(self)
    end

    # Computes the maximum value of a Tensor
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # max(v) # => 4
    # ```
    def max
      Statistics.max(self)
    end

    # Computes the maximum value of a Tensor
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # max(v) # => 4
    # ```
    def max(axis : Int32)
      Statistics.max(self, axis)
    end

    # Computes the minimum value of a Tensor
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # min(v) # => 1
    # ```
    def min
      Statistics.min(self)
    end

    # Computes the minimum value of a Tensor
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # min(v) # => 1
    # ```
    def min(axis : Int32)
      Statistics.min(self, axis)
    end

    # Computes the "peak to peak" of a Tensor (max - min)
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # v.ptp # => 3
    # ```
    def ptp
      max - min
    end

    # Computes the "peak to peak" of a Tensor (max - min)
    #
    # ```
    # v = Tensor.new [1, 2, 3, 4]
    # v.ptp # => 3
    # ```
    def ptp(axis : Int32)
      max(axis) - min(axis)
    end
  end
end
