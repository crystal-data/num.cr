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

require "stbimage"

module Num::IO
  extend self

  private def read_image_channels(filename : String, desired_channels : Int32) : Tensor(UInt8)
    width, height, channels = 0, 0, 0

    raw = LibStbImage.load(filename, pointerof(width), pointerof(height), pointerof(channels), desired_channels)

    if raw.null?
      raise String.new LibStbImage.failure_reason
    end

    return raw.to_slice(width * height * channels).to_tensor.reshape(width, height, channels).transpose(2, 0, 1)
  end

  private def read_image_channels_hw(filename : String, desired_channels : Int32, new_width : Int32, new_height : Int32) : Tensor(UInt8)
    width, height, channels = 0, 0, 0

    raw = LibStbImage.load(filename, pointerof(width), pointerof(height), pointerof(channels), desired_channels)

    if raw.null?
      raise String.new LibStbImage.failure_reason
    end

    new_buffer = Pointer(UInt8).malloc(new_width * new_height * desired_channels)

    LibStbImage.resize_uint8(raw, width, height, 0, new_buffer, new_width, new_height, 0, desired_channels)

    new_buffer.to_slice(new_width * new_height * desired_channels).to_tensor.reshape(new_width, new_height, channels).transpose(2, 0, 1)
  end

  def read_image(filename : String) : Tensor(UInt8)
    read_image_channels(filename, 0)
  end

  def read_image_grayscale(filename : String) : Tensor(UInt8)
    read_image_channels(filename, 1)
  end

  def read_image_grayscale_resize(filename : String, width : Int32, height : Int32) : Tensor(UInt8)
    read_image_channels_hw(filename, 1, width, height)
  end

  macro write_img_impl(fn_name)
    def {{fn_name}}(img : Tensor(UInt8), filename : String)
      img = img.dup(Num::RowMajor)
      w, h, c = img.shape
      success = LibStbImage.{{ fn_name }}(filename, w, h, c, img.to_unsafe, 0)

      unless success
        raise "Write failed"
      end
    end
  end

  write_img_impl write_png
  write_img_impl write_bmp
  write_img_impl write_tga
end
