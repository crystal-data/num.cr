require "../../__test__"

macro test_elementwise(first, second, output, operation, description)
  it {{description}} do
    a = Tensor.from_array {{first}}
    b = Tensor.from_array {{second}}
    res = B.{{operation}}(a, b)
    assert_array_equal res, Tensor.from_array {{output}}
  end
end

macro test_trig(first, operations)
  {% for operation in operations %}
    it "applies the stdlibs {{operation}} operation to a tensor" do
      desired = Tensor.from_array({{first}}.map { |el| Math.{{operation}}(el) })
      inp = Tensor.from_array {{first}}
      res = B.{{operation}}(inp)
      assert_array_equal res, desired
    end
  {% end %}
end

describe Bottle::BMath do
  describe "elementwise" do
    test_elementwise [1, 2], [3, 4], [4, 6], add, "adds two similarly shaped tensors"
    test_elementwise [1, 2], [3, 4], [-2, -2], subtract, "subtracts two similarly shaped tensors"
    test_elementwise [1, 2], [3, 4], [3, 8], multiply, "multiplies two similarly shaped tensors"
    test_elementwise [1, 2], [2, 4], [0.5, 0.5], divide, "divides two similarly shaped tensors"
    test_elementwise [1, 2], [2, 2], [1, 4], power, "raises a tensor to the power of another tensor"
    test_elementwise [4, 4], [3, 3], [1, 1], floordiv, "floor divides two similarly shaped tensors"
    test_elementwise [2, 2], [2, 2], [0, 0], modulo, "remainder of two similarly shaped tensors"
    test_elementwise [1, 2], [1, 3], [true, false], equal, "checks equality of two tensors"
    test_elementwise [1, 2], [1, 1], [false, true], greater, "checks greater than of two tensors"
    test_elementwise [1, 2], [1, 1], [true, true], greater_equal, "checks greater equal of two tensors"
    test_elementwise [1, 2], [1, 3], [false, true], less, "checks less than for two tensors"
    test_elementwise [1, 2], [1, 3], [true, true], less_equal, "checks less equal for two tensors"
    test_elementwise [true, false], [false, true], [false, false], bitwise_and, "bitwise and of two tensors"
    test_elementwise [true, true], [false, false], [true, true], bitwise_or, "bitwise or of two tensors"
    test_elementwise [true, false], [false, false], [true, false], bitwise_xor, "bitwise xor of two tensors"
    test_elementwise [1, 2], [2, 2], [4, 8], left_shift, "left shifts two tensors"
    test_elementwise [1, 2], [2, 2], [0, 0], right_shift, "right shifts two tensors"
  end

  describe "elementwise_trig" do
    test_trig [1, 2, 3, 4, 5], [
      acos, acosh, asin, asinh, atan, atanh, besselj0, besselj1, bessely0,
      bessely1, cbrt, cos, cosh, erf, erfc, exp, exp2, expm1, gamma, ilogb,
      lgamma, log, log10, log1p, log2, logb, sin, sinh, sqrt, tan, tanh
    ]
  end
end
