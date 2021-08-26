# Copyright (c) 2021 Crystal Data Contributors
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

module Num::Exceptions
  # Raised when a sequence subscript is out of range.
  # (Slice indices are silently truncated to fall in
  # the allowed range; if an index is not an integer,
  # TypeCastError is raised.)
  class IndexError < Exception
  end

  # Raised when an operation or function receives an
  # argument that has the right type but an inappropriate
  # value, and the situation is not described by a more
  # precise exception such as IndexError.
  class ValueError < Exception
  end

  # This is raised whenever an axis parameter is specified
  # that is larger than the number of array dimensions.
  class AxisError < Exception
  end

  # Generic exception raised by linalg functions.
  # General purpose exception class, programmatically
  # raised in linalg functions when a Linear
  # Algebra-related condition would prevent further
  # correct execution of the function.
  class LinAlgError < Exception
  end
end
