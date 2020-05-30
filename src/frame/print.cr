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

require "./index"
require "./lane"
require "../array/print"
require "../num"

module Frame
  extend self

  def print_lane(io, lane : Lane(U, V), padding = 3) forall U, V
    index_width = NumInternal.max_width(lane.index.data)
    data_width = NumInternal.max_width(lane.data)
    s = ""
    lane.each_with_index do |e, i|
      io << "#{NumInternal.format(i).ljust(padding + index_width)}"
      io << "#{NumInternal.format(e).rjust(data_width)}"
      io << "\n"
    end
    io << "dtype: #{U}\n"
    io << "index : #{lane.index.class}"
    io
  end
end
