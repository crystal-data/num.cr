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

alias UInt8Tensor = Tensor(UInt8, CPU(UInt8))
alias UInt16Tensor = Tensor(UInt16, CPU(UInt16))
alias UInt32Tensor = Tensor(UInt16, CPU(UInt16))
alias UInt64Tensor = Tensor(UInt64, CPU(UInt64))

alias Int8Tensor = Tensor(Int8, CPU(Int8))
alias Int16Tensor = Tensor(Int16, CPU(Int16))
alias Int32Tensor = Tensor(Int32, CPU(Int32))
alias Int64Tensor = Tensor(Int64, CPU(Int64))

alias Float32Tensor = Tensor(Float32, CPU(Float32))
alias Float64Tensor = Tensor(Float64, CPU(Float64))

alias ComplexTensor = Tensor(Complex, CPU(Complex))
alias BoolTensor = Tensor(Bool, CPU(Bool))

alias Float32ClTensor = Tensor(Float32, OCL(Float32))
alias Float64ClTensor = Tensor(Float64, OCL(Float64))
