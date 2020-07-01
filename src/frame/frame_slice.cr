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

require "./frame"

# A `FrameSlice` is a row copy of a `DataFrame`.  It owns its
# own data, and cannot be modified.  It is the result of a
# row-wise slice of a `DataFrame`.
#
# When used it an operation against a `DataFrame`, it broadcasts
# against the columns of a `DataFrame`, so that reductions will
# always be able to be operated upon
class FrameSlice(T)
  getter c : T

  # Initializes a DataFrame from a variadic number of arguments.
  # This should only be used by the `DataFrame` class, but remains
  # public
  def initialize(**args : **T)
    @c = args
  end

  # :nodoc:
  def to_s(io)
    kw = 0
    vw = 0
    @c.each do |k, v|
      ks = "#{k}".size
      vs = Num::Internal.format(v).size
      if ks > kw
        kw = ks
      end
      if vs > vw
        vw = vs
      end
    end
    @c.each do |k, v|
      io << "#{k}".rjust(kw)
      io << "  "
      io << Num::Internal.format(v).rjust(vw)
      io << "\n"
    end
  end
end
