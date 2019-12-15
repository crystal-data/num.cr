require "../spec_helper"

macro test_numerical_extension(arr, value, dtypes, operators)
  {% for operator in operators %}
    {% for dtype in dtypes %}
      it "checks the rhs {{operator}} of number->tensor" do
          m = Tensor.from_array({{arr}})
          expected = {{arr}}.map { |e| {{value}} {{operator.id}} e }
          result = {{value}} {{operator.id}} m
          assert_array_equal Tensor.from_array(expected), result
      end
    {% end %}
  {% end %}
end

macro test_complex_extension(arr, value, operators)
  {% for operator in operators %}
    it "checks the rhs {{operator}} of number->tensor" do
      inp = {{arr}}.map { |e| Complex.new(e, e) }
      m = Tensor.from_array(inp)
      lhs = Complex.new({{value}}, {{value}})
      expected = inp.map { |e| lhs {{operator.id}} e }
      result = lhs {{operator.id}} m
      assert_array_equal Tensor.from_array(expected), result
    end
  {% end %}
end

describe Num::BaseArray do
  describe "BaseArray#NumberExtensions" do
    test_numerical_extension [1, 2, 3, 4], 3, [
      Float32,
      Float64,
      Int16,
      Int32,
      Int64,
      Int8,
      UInt16,
      UInt32,
      UInt64,
      UInt8,
    ], [:+, :-, :*, :/, ://, :**, :%, :==, :>, :>=, :<, :<=]

    test_complex_extension [1, 2, 3, 4], 3, [
      :+, :-, :*, :/, :==,
    ]
  end
end
