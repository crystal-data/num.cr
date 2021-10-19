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

# :nodoc:
# A Node is a member of a computational graph that contains
# a reference to a gate, as well as the parents of the operation
# and the payload that resulted from the operation.
class Num::Grad::Node(T)
  # A Gate containing a backwards and cache function for
  # a node
  getter gate : Num::Grad::Gate(T)

  # The variables that created this node
  getter parents : Array(Num::Grad::Variable(T))

  # Wrapper around a Tensor, contains operation data
  getter payload : Num::Grad::Payload(T)

  # Debug use only, contains a name for a node
  getter name : String

  # This initializer shouldn't ever be called outside of
  # Num::Grad.register.
  #
  # Users defining custom gradients and gates should
  # follow the same rule
  def initialize(
    @gate : Gate(T),
    @parents : Array(Num::Grad::Variable(T)),
    @payload : Num::Grad::Payload(T),
    @name : String = ""
  )
  end
end
