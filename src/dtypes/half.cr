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

struct Float16
  @value : UInt16 = 0

  def initialize(n : Float32)
    @value = f32_to_f16(n)
  end

  def initialize(n : Float64)
    @value = f64_to_f16(n)
  end

  private def f32_to_f16(value : Float32) : UInt16
    x = value.unsafe_as(UInt32)

    sign = x & 0x8000_0000_u32
    exp = x & 0x7F80_0000_u32
    man = x & 0x007F_FFFF_u32

    if exp == 0x7F80_0000_u32
      nan_bit = man == 0 ? 0 : 0x0200_u32
      return ((sign >> 16) | 0x7C00u32 | nan_bit | (man >> 13)).to_u16
    end

    half_sign = sign >> 16
    unbiased_exp = ((exp >> 23).to_i) - 127
    half_exp = unbiased_exp + 15

    if half_exp >= 0x1F
      return (half_sign | 0x7C00u32).to_u16
    end

    if half_exp <= 0
      if 14 - half_exp > 24
        return half_sign.to_u16
      end
      man = man | 0x0080_0000_u32
      half_man = man >> (14 - half_exp)
      round_bit = 1 << (13 - half_exp)
      if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0
        half_man += 1
      end
      return (half_sign | half_man).to_u16
    end

    half_exp = (half_exp.to_u32) << 10
    half_man = man >> 13
    round_bit = 0x0000_1000_u32
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0
      ((half_sign | half_exp | half_man) + 1).to_u16
    else
      (half_sign | half_exp | half_man).to_u16
    end
  end

  private def f64_to_f16(value : Float64) : UInt16
    val = value.unsafe_as(UInt64)
    x = (val >> 32).to_u32

    sign = x & 0x8000_0000_u32
    exp = x & 0x7FF0_0000_u32
    man = x & 0x000F_FFFF_u32

    if exp == 0x7FF0_0000_u32
      nan_bit = man == 0 && (val.to_u32 == 0) ? 0 : 0x0200_u32
      return ((sign >> 16) | 0x7C00_u32 | nan_bit | (man >> 10)).to_u16
    end

    half_sign = sign >> 16
    unbiased_exp = ((exp >> 20).to_i64) - 1023
    half_exp = unbiased_exp + 15

    if half_exp >= 0x1F
      return (half_sign | 0x7C00u32).to_u16
    end

    if half_exp <= 0
      if 10 - half_exp > 21
        return half_sign.to_u16
      end
      man = man | 0x0010_0000_u32
      half_man = man >> (11 - half_exp)
      round_bit = 1 << (10 - half_exp)
      if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0
        half_man += 1
      end
      return (half_sign | half_man).to_u16
    end

    half_exp = (half_exp.to_u32) << 10
    half_man = man >> 10
    round_bit = 0x0000_0200_u32
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0
      ((half_sign | half_exp | half_man) + 1).to_u16
    else
      (half_sign | half_exp | half_man).to_u16
    end
  end

  def to_f32 : Float32
    if @value & 0x7FFF_u16 == 0
      return (@value.to_u32 << 16).unsafe_as(Float32)
    end

    half_sign = (@value & 0x8000u16).to_u32
    half_exp = (@value & 0x7C00u16).to_u32
    half_man = (@value & 0x03FFu16).to_u32

    if half_exp == 0x7C00u32
      if half_man == 0
        return ((half_sign << 16) | 0x7F80_0000_u32).unsafe_as(Float32)
      else
        return ((half_sign << 16) | 0x7FC0_0000_u32 | (half_man << 13)).unsafe_as(Float32)
      end
    end

    sign = half_sign << 16
    unbiased_exp = ((half_exp.to_i) >> 10) - 15

    if half_exp == 0
      e = (half_man.to_u16).leading_zeros_count - 6

      exp = (127 - 15 - e) << 23
      man = (half_man << (14 + e)) & 0x7F_FF_FFu32
      return (sign | exp | man).unsafe_as(Float32)
    end

    exp = ((unbiased_exp + 127).to_u32) << 23
    man = (half_man & 0x03FF_u32) << 13
    (sign | exp | man).unsafe_as(Float32)
  end

  def to_f : Float64
    if @value & 0x7FFF_u16 == 0
      return ((@value.to_u64) << 48).unsafe_as(Float64)
    end

    half_sign = (@value & 0x8000u16).to_u64
    half_exp = (@value & 0x7C00u16).to_u64
    half_man = (@value & 0x03FFu16).to_u64

    if half_exp == 0x7C00_u64
      if half_man == 0
        return ((half_sign << 48) | 0x7FF0_0000_0000_0000u64).unsafe_as(Float64)
      else
        return ((half_sign << 48) | 0x7FF8_0000_0000_0000_u64 | (half_man << 42)).unsafe_as(Float64)
      end
    end

    sign = half_sign << 48
    unbiased_exp = ((half_exp.to_i64) >> 10) - 15

    if half_exp == 0
      e = (half_man.to_u64).leading_zeros_count - 6

      exp = ((1023 - 15 - e).to_u64) << 52
      man = (half_man << (43 + e)) & 0xF_FFFF_FFFF_FFFF_u64
      return (sign | exp | man).unsafe_as(Float64)
    end

    exp = ((unbiased_exp + 1023).to_u64) << 52
    man = (half_man & 0x03FF_u64) << 42
    (sign | exp | man).unsafe_as(Float64)
  end

  def to_s(io)
    io << self.to_f
  end
end
