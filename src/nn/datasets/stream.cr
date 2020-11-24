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

class Num::NN::DataStream
  getter shuffle : Bool
  getter batch_size : Int32
  getter width : Int32
  getter height : Int32
  getter num_classes : Int32
  getter mapping : Array(Tuple(String, Int32))

  private def initialize(mapping : Hash(String, Int32), @width, @height, @num_classes, @shuffle, @batch_size)
    @mapping = Array(Tuple(String, Int32)).new(mapping.size)
    mapping.each do |k, v|
      @mapping << {k, v}
    end

    if shuffle
      @mapping.shuffle!
    end
  end

  def self.build(path : String, width : Int32, height : Int32, num_classes : Int32, shuffle : Bool = true, batch_size : Int32 = 32) : Num::NN::DataStream
    raise "Build must be implemented"
  end

  def each
    chunks = @mapping.size // @batch_size

    chunks.times do |i|
      offset = i * batch_size
      chunk = @mapping[offset...offset + batch_size]

      data = Tensor(Float32).new([@batch_size, 1, @width, @height])
      truth = Tensor(Float32).new([@batch_size, @num_classes])

      chunk.each_with_index do |el, index|
        fn, cls = el
        img = Num::IO.read_image_grayscale(fn)
        data[index] = img / 255
        t = (0...@num_classes).map do |i|
          i == cls ? 1 : 0
        end
        truth[index] = t.to_tensor
      end
      yield data, truth
    end
  end
end
