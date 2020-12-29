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

module Num::Backend
  extend self

  def tensor_to_crystal_array(device : CPU(U)) : Array(U) forall U
    a = Array(U).new(device.size)
    each(device) do |el|
      a << el
    end
    a
  end

  def cast_tensor(device : CPU(U), dtype : V.class) : Tensor(V) forall U, V
    casted = Tensor(V).new(device.shape, device: CPU(V))
    map!(casted.storage, device) do |_, j|
      j
    end
    casted
  end

  def copy_tensor(device : CPU(U), order : Num::OrderType) : Tensor(U) forall U
    copied = Tensor(U).new(device.shape, order: order, device: CPU(U))
    map!(copied.storage, device) do |_, j|
      j
    end
    copied
  end
end
