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

macro init_csr_iteration(prefix, dtype)
  {{ prefix }}_count = 0
  {{ prefix }}_next = {{ prefix }}.cols.size == 0 ? 0 : {{ prefix }}.cols[0]
  {{ prefix }}_zero = {{ dtype }}.new(0)
end

macro init_csc_iteration(prefix, dtype)
  {{ prefix }}_count = 0
  {{ prefix }}_next = {{ prefix }}.rows.size == 0 ? 0 : {{ prefix }}.rows[0]
  {{ prefix }}_zero = {{ dtype }}.new(0)
end

macro advanced_csr_iteration(prefix)
  if {{ prefix }}_count < {{ prefix }}_max && j == {{ prefix }}_next
    {{ prefix }}_val = {{ prefix }}.vals[{{ prefix }}_count]
    {{ prefix }}_count += 1
    if {{ prefix }}_count < {{ prefix }}.nnz
      {{ prefix }}_next = {{ prefix }}.cols[{{ prefix }}_count]
    end
  else
    {{ prefix }}_val = {{ prefix }}_zero
  end
end

macro advanced_csc_iteration(prefix)
  if {{ prefix }}_count < {{ prefix }}_max && i == {{ prefix }}_next
    {{ prefix }}_val = {{ prefix }}.vals[{{ prefix }}_count]
    {{ prefix }}_count += 1
    if {{ prefix }}_count < {{ prefix }}.nnz
      {{ prefix }}_next = {{ prefix }}.rows[{{ prefix }}_count]
    end
  else
    {{ prefix }}_val = {{ prefix }}_zero
  end
end

macro add_csr_vals
  unless result == 0
    n += 1
    new_vals << result
    new_cols << j
  end
end

macro add_csc_vals
  unless result == 0
    n += 1
    new_vals << result
    new_rows << i
  end
end

macro add_csr_rows
  new_rows << n
end

macro add_csc_rows
  new_cols << n
end
