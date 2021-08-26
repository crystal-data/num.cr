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

@[Flags]
enum Num::ArrayFlags
  # Contiguous really means C-style contiguious.  The
  # contiguous part means that there are no 'skipped
  # elements'.  That is, that a flat_iter over the array will
  # touch every location in memory from the location of the
  # first element to that of the last element.  The C-style
  # part means that the data is laid out such that the last index
  # is the fastest varying as one scans though the array's
  # memory.
  Contiguous
  # Fortran really means Fortran-style contiguious.  The
  # contiguous part means that there are no 'skipped
  # elements'.  That is, that a flat_iter over the array will
  # touch every location in memory from the location of the
  # first element to that of the last element.  The Fortran-style
  # part means that the data is laid out such that the first index
  # is the fastest varying as one scans though the array's
  # memory.
  Fortran
  # OwnData indicates if this array is the owner of the data
  # pointed to by its .ptr property.  If not then this is a
  # view onto some other array's data.
  OwnData
  # Some views into arrays are created using stride tricks
  # and aren't safe to write to, since many locations may
  # be sharing the same memory
  Write
end
