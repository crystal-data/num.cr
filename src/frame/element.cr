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

module Frame::Element(T)
  abstract def set(value : U) forall U
  abstract def ==(value : Frame::Element)
  abstract def !=(value : Frame::Element)
  abstract def <(value : Frame::Element)
  abstract def <=(value : Frame::Element)
  abstract def >(value : Frame::Element)
  abstract def >=(value : Frame::Element)
  abstract def copy : Frame::Element
  abstract def val : T
  abstract def to_s(io)
  abstract def to_i : Int32
  abstract def to_f : Float64
  abstract def to_bool : Bool
  abstract def is_na : Bool
  abstract def dtype : T.class
end

struct FBool
  include Frame::Element(Bool)
  @e : Bool
  @nan : Bool = false

  def initialize(@e : Bool, @nan : Bool = false)
  end

  def set(value : U) forall U
    @nan = false
    case U
    when String
      case value.lower
      when "nan"
        @nan = true
      when "true", "t", "1"
        @e = true
      when "false", "f", "0"
        @e = false
      else
        @nan = true
      end
    when Int32
      case value
      when 0
        @e = false
      else
        @e = true
      end
    when Float64
      if value.nan?
        @nan = true
      elsif value == 0
        @e = false
      else
        @e = true
      end
    when Bool
      @e = value
    when Frame::Element
      @e = value.to_bool
      @nan = value.is_na
    else
      @nan = true
    end
  end

  def copy
    self.class.new(@e, @nan)
  end

  def dtype
    Bool
  end

  def to_s(io)
    if is_na
      io << "NaN"
    elsif @e
      io << "true"
    else
      io << "false"
    end
  end
end
