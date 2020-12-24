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

  # Turns a standard library Crystal pointer into a storage instance
  #
  # Requires a shape, memory layout, as well as a device so the
  # correct overload is chosen
  @[AlwaysInline]
  def hostptr_to_storage(ptr : Pointer(U), shape : Array(Int), order : Num::OrderType, device : CPU.class) : CPU(U) forall U
    CPU(U).new(ptr, shape, order)
  end

  # Turns a standard library Crystal array into a storage instance
  #
  # Requires a shape, as well as a device so the
  # correct overload is chosen
  @[AlwaysInline]
  def flat_array_to_storage(a : Array(U), shape, device : CPU.class) forall U
    CPU(U).new(a.to_unsafe, shape)
  end
end
