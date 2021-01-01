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

class Tensor(T, S)
  getter data : S

  # Returns the size of a Tensor along each dimension
  #
  # ```
  # a = Tensor(UInt8).new([2, 3, 4])
  # a.shape # => [2, 3, 4]
  # ```
  getter shape : Array(Int32)

  # Returns the step of a Tensor along each dimension
  #
  # ```
  # a = Tensor(UInt8).new([3, 3, 2])
  # a.shape # => [4, 2, 1]
  # ```
  getter strides : Array(Int32)

  # Returns the offset of a Tensor's data
  #
  # ```
  # a = Tensor(UInt8).new([2, 3, 4])
  # a.offset # => 0
  # ```
  getter offset : Int32

  # Returns the size of a Tensor along each dimension
  #
  # ```
  # a = Tensor(UInt8).new([2, 3, 4])
  # a.shape # => [2, 3, 4]
  # ```
  getter size : Int32

  # Returns the number of dimensions in a Tensor
  #
  # ```
  # a = Tensor(UInt8).new([3, 3, 3, 3])
  # a.rank # => 4
  # ```
  def rank : Int32
    @shape.size
  end

  def to_s(io)
    io << to_s
  end

  def to_s : String
    Num::Internal.array_to_string(self)
  end

  private macro delegate_to_backend(method)
    def {{method.id}}(*args, **options)
      Num.{{method.id}}(self, *args, **options)
    end
  end

  private macro alias_to_backend(method, alias_name, both = true)
    {% if both %}
      def {{method.id}}(*args, **options)
        Num.{{method.id}}(self, *args, **options)
      end
    {% end %}

    def {{alias_name.id}}(*args, **options)
      Num.{{method.id}}(self, *args, **options)
    end
  end

  private macro assert_types
    {% if T != S.type_vars[0] %}
      {% raise "A Tensor and it's storage must share the same dtype" %}
    {% end %}
  end

  # :nodoc:
  def is_f_contiguous : Bool
    return true unless self.rank != 0
    if self.rank == 1
      return @shape[0] == 1 || @strides[0] == 1
    end
    s = 1
    self.rank.times do |i|
      d = @shape[i]
      return true unless d != 0
      return false unless @strides[i] == s
      s *= d
    end
    true
  end

  # :nodoc:
  def is_c_contiguous : Bool
    return true unless self.rank != 0
    if self.rank == 1
      return @shape[0] == 1 || @strides[0] == 1
    end

    s = 1
    (self.rank - 1).step(to: 0, by: -1) do |i|
      d = @shape[i]
      return true unless d != 0
      return false unless @strides[i] == s
      s *= d
    end
    true
  end
end
