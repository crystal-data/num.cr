require "../spec_helper"

macro test_broadcast(ashape, bshape, oshape)
  it "broadcasts {{ashape}} and {{bshape}} to {{oshape}}" do
    a = MockArray(Int32).new({{ashape}})
    b = MockArray(Int32).new({{bshape}})
    a.broadcastable(b).should eq {{oshape}}
  end
end

macro test_bad_broadcast(ashape, bshape)
  it "raises shape error on broadcast between {{ashape}} and {{bshape}}" do
    a = MockArray(Int32).new({{ashape}})
    b = MockArray(Int32).new({{bshape}})
    expect_raises(ShapeError) do
      a.broadcastable(b)
    end
  end
end

describe Num::BaseArray do
  describe "Num#broadcast_to" do
    it "broadcasts values for 1d array" do
      m = MockArray.new([3]) { |i| i }
      b = m.broadcast_to([3, 3])
      expected = MockArray.from_array [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
      assert_array_equal expected, b
    end

    it "broadcasts column array correctly" do
      m = MockArray.new([3, 1]) { |i| i }
      b = m.broadcast_to([3, 3])
      expected = MockArray.from_array [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
      assert_array_equal expected, b
    end

    it "broadcast_to raises shape error on bad broadcast" do
      m = MockArray.new([3]) { |i| i }
      expect_raises(ShapeError) do
        m.broadcast_to([5, 5])
      end
    end
  end

  describe "Num#broadcastable" do
    it "returns an empty shape for arrays that already have the same shape" do
      m = MockArray.new([2, 2]) { |i| i }
      shape = m.broadcastable(m)
      shape.should eq [] of Int32
    end

    test_broadcast [256, 256, 3], [3], [256, 256, 3]
    test_broadcast [8, 1, 6, 1], [7, 1, 5], [8, 7, 6, 5]
    test_broadcast [5, 4], [1], [5, 4]
    test_broadcast [15, 3, 5], [15, 1, 5], [15, 3, 5]
    test_broadcast [15, 3, 5], [3, 5], [15, 3, 5]
    test_broadcast [15, 3, 5], [3, 1], [15, 3, 5]

    test_bad_broadcast [3], [4]
    test_bad_broadcast [2, 1], [8, 4, 3]
  end

  describe "Num#as_strided" do
    it "creates a valid strided rolling view" do
      m = MockArray.new([8]) { |i| i }
      n = m.strides[0]
      res = m.as_strided([5, 3], [n, n])
      expected = MockArray.from_array [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
      assert_array_equal res, expected
    end
  end
end
