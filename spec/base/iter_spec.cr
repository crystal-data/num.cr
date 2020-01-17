# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

require "../spec_helper"

require "../spec_helper"

describe Num::BaseArray do
  describe "BaseArray#safeiters" do
    # it "contiguous array returns contig iter" do
    #   m = Num::BaseArray.new([3, 3]) { |i| i }
    #   m.flat_iter.is_a?(SafeFlat).should be_true
    # end
    #
    # it "noncontig array returns nd iter" do
    #   m = Num::BaseArray.new([3, 3]) { |i| i }
    #   m[..., 1].flat_iter.is_a?(SafeND).should be_true
    # end

    it "contig iter returns right values" do
      m = Num::BaseArray.new([2, 2]) { |i| i }
      expected = [] of Int32
      m.iter.each { |e| expected << e.value }
      expected.should eq [0, 1, 2, 3]
    end

    it "nd iter returns the right values" do
      m = Num::BaseArray.new([2, 2]) { |i| i }
      res = [] of Int32
      m[..., 1].iter.each { |e| res << e.value }
      res.should eq [1, 3]
    end
  end
end
