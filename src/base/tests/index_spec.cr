require "../../__test__"

describe Num::BaseArray do
  describe "BaseArray#indexing" do
    it "accesses scalar value" do
      m = MockArray.new([3, 2, 2]) { |i| i }
      m[0, 0, 1].value.should eq 1
    end

    it "identifies scalar value out of range" do
      m = MockArray.new([4]) { |i| i }
      expect_raises(IndexError) do
        m[6]
      end
    end

    it "valid conversion of negative indices" do
      m = MockArray.new([3]) { |i| i }
      m[-1].value.should eq 2
    end

    it "set a scalar value" do
      m = MockArray.new([2, 2]) { |i| i }
      m[0, 0] = 100
      expected = MockArray.from_array [[100, 1], [2, 3]]
      assert_array_equal m, expected
    end

    it "set a scalar value of range raises" do
      m = MockArray.new([3]) { |i| i }
      expect_raises(IndexError) do
        m[10] = 5
      end
    end

    it "sets based on negative indices" do
      m = MockArray.new([3]) { |i| i }
      m[-1] = 100
      expected = MockArray.from_array [0, 1, 100]
      assert_array_equal expected, m
    end

    it "accesses a slice of an nd array" do
      m = MockArray.new([3, 2, 2]) { |i| i }
      slice = m[0]
      expected = MockArray.from_array [[0, 1], [2, 3]]
      assert_array_equal slice, expected
    end

    it "accesses strided slice of an nd array" do
      m = MockArray.new([3, 2, 2]) { |i| i }
      slice = m[..., 0]
      expected = MockArray.from_array [[0, 1], [4, 5], [8, 9]]
      assert_array_equal slice, expected
    end
  end
end
