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

class Tensor(T, S)
  # Yields the elements of a `Tensor`, always in RowMajor order,
  # as if the `Tensor` was flat.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # a.each do |el|
  #   puts el
  # end
  #
  # # 0
  # # 1
  # # 2
  # # 3
  # ```
  def each
    Num.each(self) do |el|
      yield el
    end
  end

  # Yields the elements of two `Tensor`s, always in RowMajor order,
  # as if the `Tensor`s were flat.
  #
  # Arguments
  # ---------
  # *b* : Tensor
  #   The other tensor to iterate along
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # b = Tensor.new(2, 2) { |i| i + 2 }
  # a.zip(b) do |el|
  #   puts el
  # end
  #
  # # { 0, 2}
  # # { 1, 3}
  # # { 2, 4}
  # # { 3, 5}
  # ```
  @[AlwaysInline]
  def zip(b : Tensor(U, CPU(U)), &block : T, U -> _) forall U, V
    Num.zip(self, b) do |i, j|
      yield i, j
    end
  end

  # Yields the elements of a `Tensor` lazily, always in RowMajor order,
  # as if the `Tensor` was flat.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # iter = a.each
  # a.size.times do
  #   puts iter.next.value
  # end
  #
  # # 0
  # # 1
  # # 2
  # # 3
  # ```
  def each
    Num.each(self)
  end

  # Yields the memory locations of each element of a `Tensor`, always in
  # RowMajor oder, as if the `Tensor` was flat.
  #
  # This should primarily be used by internal methods.  Methods such
  # as `map!` provided more convenient access to editing the values
  # of a `Tensor`
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # a.each_pointer do |el|
  #   puts el.value
  # end
  #
  # # 0
  # # 1
  # # 2
  # # 3
  # ```
  def each_pointer
    Num.each_pointer(self) do |el|
      yield el
    end
  end

  # Yields the elements of a `Tensor`, always in RowMajor order,
  # as if the `Tensor` was flat.  Also yields the flat index of each
  # element.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # a.each_with_index do |el, i|
  #   puts "#{el}_#{i}"
  # end
  #
  # # 0_0
  # # 1_1
  # # 2_2
  # # 3_3
  # ```
  def each_with_index
    Num.each_with_index(self) do |el, i|
      yield el, i
    end
  end

  # Yields the memory locations of each element of a `Tensor`, always in
  # RowMajor oder, as if the `Tensor` was flat.  Also yields the flat
  # index of a `Tensor`
  #
  # This should primarily be used by internal methods.  Methods such
  # as `map!` provided more convenient access to editing the values
  # of a `Tensor`
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # a.each_pointer_with_index do |el, i|
  #   puts "#{el.value}_#{i}"
  # end
  #
  # # 0_0
  # # 1_1
  # # 2_2
  # # 3_3
  # ```
  def each_pointer_with_index
    Num.each_pointer_with_index(self) do |el, i|
      yield el, i
    end
  end

  # Maps a block across a `Tensor`.  The `Tensor` is treated
  # as flat during iteration, and iteration is always done
  # in RowMajor order
  #
  # The generic type of the returned `Tensor` is inferred from
  # the block
  #
  # Arguments
  # ---------
  # *block* Proc(T, U)
  #   Proc to map across the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # a.map { |e| e + 5 } # => [5, 6, 7]
  # ```
  def map(&block : T -> U) forall U
    Num.map(self) do |el|
      yield el
    end
  end

  # Maps a block across a `Tensor` in place.  The `Tensor` is treated
  # as flat during iteration, and iteration is always done
  # in RowMajor order
  #
  # Arguments
  # ---------
  # *block* Proc(T, U)
  #   Proc to map across the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # a.map! { |e| e + 5 }
  # a # => [5, 6, 7]
  # ```
  def map!(&block)
    Num.map!(self) do |el|
      yield el
    end
  end

  # Maps a block across two `Tensors`.  This is more efficient than
  # zipping iterators since it iterates both `Tensor`'s in a single
  # call, avoiding overhead from tracking multiple iterators.
  #
  # The generic type of the returned `Tensor` is inferred from a block
  #
  # Arguments
  # ---------
  # *t* : Tensor(U)
  #   The second `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self`
  # *block* : Proc(T, U, V)
  #   The block to map across both `Tensor`s
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # b = Tensor.new([3]) { |i| i }
  #
  # a.map(b) { |i, j| i + j } # => [0, 2, 4]
  # ```
  def map(other : Tensor, &block)
    Num.map(self, other) do |i, j|
      yield i, j
    end
  end

  # Maps a block across two `Tensors`.  This is more efficient than
  # zipping iterators since it iterates both `Tensor`'s in a single
  # call, avoiding overhead from tracking multiple iterators.
  # The result of the block is stored in `self`.
  #
  # Broadcasting rules still apply, but since this is an in place
  # operation, the other `Tensor` must broadcast to the shape of `self`
  #
  # Arguments
  # ---------
  # *t* : Tensor(U)
  #   The second `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self`
  # *block* : Proc(T, U, T)
  #   The block to map across both `Tensor`s
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # b = Tensor.new([3]) { |i| i }
  #
  # a.map!(b) { |i, j| i + j }
  # a # => [0, 2, 4]
  # ```
  def map!(other : Tensor, &block)
    Num.map!(self, other) do |i, j|
      yield i, j
    end
  end

  # Maps a block across three `Tensors`.  This is more efficient than
  # zipping iterators since it iterates all `Tensor`'s in a single
  # call, avoiding overhead from tracking multiple iterators.
  #
  # The generic type of the returned `Tensor` is inferred from a block
  #
  # Arguments
  # ---------
  # *t* : Tensor(U)
  #   The second `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self` and `v`
  # *v) : Tensor(V)
  #   The third `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self` and `t`
  # *block* : Proc(T, U, V, W)
  #   The block to map across all `Tensor`s
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # b = Tensor.new([3]) { |i| i }
  # c = Tensor.new([3]) { |i| i }
  #
  # a.map(b, c) { |i, j, k| i + j + k } # => [0, 3, 6]
  # ```
  def map(d1 : Tensor, d2 : Tensor, &block)
    Num.map(self, d1, d2) do |i, j, k|
      yield i, j, k
    end
  end

  # Maps a block across three `Tensors`.  This is more efficient than
  # zipping iterators since it iterates all `Tensor`'s in a single
  # call, avoiding overhead from tracking multiple iterators.
  # The result of the block is stored in `self`.
  #
  # Broadcasting rules still apply, but since this is an in place
  # operation, the other `Tensor`'s must broadcast to the shape of `self`
  #
  # Arguments
  # ---------
  # *t* : Tensor(U)
  #   The second `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self` and `v`
  # *v) : Tensor(V)
  #   The third `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self` and `t`
  # *block* : Proc(T, U, V, W)
  #   The block to map across all `Tensor`s
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # b = Tensor.new([3]) { |i| i }
  # c = Tensor.new([3]) { |i| i }
  #
  # a.map!(b, c) { |i, j, k| i + j + k }
  # a # => [0, 3, 6]
  # ```
  def map!(d1 : Tensor, d2 : Tensor, &block)
    Num.map!(self, d1, d2) do |i, j, k|
      yield i, j, k
    end
  end

  # Combines all elements in the Tensor by applying a binary operation,
  # specified by a block, so as to reduce them to a single value.
  #
  # For each element in the Tensor the block is passed an accumulator value
  # (*memo*) and the element. The result becomes the new value for *memo*.
  # At the end of the iteration, the final value of *memo* is the return value
  # for the method. The initial value for the accumulator is the first element
  # in the Tensor.
  #
  # Raises `Enumerable::EmptyError` if the Tensor is empty.
  #
  # ```
  # [1, 2, 3, 4, 5].to_tensor.reduce { |acc, i| acc + i } # => 15
  # ```
  def reduce
    memo = uninitialized T
    found = false

    each do |elem|
      memo = found ? (yield memo, elem) : elem
      found = true
    end

    found ? memo : raise Enumerable::EmptyError.new
  end

  # Just like the other variant, but you can set the initial value of the
  # accumulator.
  #
  # ```
  # [1, 2, 3, 4, 5].to_tensor.reduce(10) { |acc, i| acc + i } # => 25
  # ```
  def reduce(memo)
    each do |elem|
      memo = yield memo, elem
    end
    memo
  end

  # Returns a Tensor containing the successive values of applying a binary
  # operation, specified by the given *block*, to this Tensor's elements.
  #
  # For each element in the Tensor the block is passed an accumulator value
  # and the element. The result becomes the new value for the accumulator and is
  # also appended to the returned Tensor. The initial value for the accumulator
  # is the first element in the Tensor.
  #
  # ```
  # [2, 3, 4, 5]..to_tensor.accumulate { |x, y| x * y } # => [2, 6, 24, 120]
  # ```
  def accumulate(&block : T, T -> T) : Tensor(T, S)
    values = Tensor(T, S).zeros([self.size])
    buffer = values.to_unsafe
    memo = uninitialized T

    each_with_index do |elem, index|
      memo = index == 0 ? elem : (yield memo, elem)
      buffer[index] = memo
    end

    values
  end

  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  def reduce_axis(axis : Int, dims : Bool = false, &block)
    Num.reduce_axis(self, axis, dims) do |i, j|
      yield i, j
    end
  end

  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  def each_axis(axis : Int, dims : Bool = false, &block)
    Num.each_axis(self, axis, dims) do |ax|
      yield ax
    end
  end

  #
  #
  #
  #
  #
  #
  #
  #
  def each_axis(axis : Int, dims : Bool = false)
    Num.each_axis(self, axis, dims)
  end

  #
  #
  #
  #
  #
  #
  #
  #
  def yield_along_axis(axis : Int)
    Num.yield_along_axis(self, axis) do |ax|
      yield ax
    end
  end
end
