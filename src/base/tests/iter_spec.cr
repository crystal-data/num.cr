require "../../__test__"

describe Bottle::BaseArray do
  describe "BaseArray#safeiters" do
    it "contiguous array returns contig iter" do
      m = MockArray.new([3, 3]) { |i| i }
      m.flat_iter.is_a?(SafeFlat).should be_true
    end

    it "noncontig array returns nd iter" do
      m = MockArray.new([3, 3]) { |i| i }
      m[..., 1].flat_iter.is_a?(SafeND).should be_true
    end

    it "contig iter returns right values" do
      m = MockArray.new([2, 2]) { |i| i }
      expected = [] of Int32
      m.flat_iter.each { |e| expected << e.value }
      expected.should eq [0, 1, 2, 3]
    end

    it "nd iter returns the right values" do
      m = MockArray.new([2, 2]) { |i| i }
      res = [] of Int32
      m[..., 1].flat_iter.each { |e| res << e.value }
      res.should eq [1, 3]
    end
  end
end
