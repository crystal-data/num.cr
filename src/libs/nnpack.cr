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

@[Link("nnpack")]
lib LibNNPACK
  enum NNPStatus : LibC::Int
    SUCCESS                           =  0
    INVALID_BATCH_SIZE                =  2
    INVALID_CHANNELS                  =  3
    INVALID_INPUT_CHANNELS            =  4
    INVALID_OUTPUT_CHANNELS           =  5
    INVALID_INPUT_SIZE                = 10
    INVALID_INPUT_STRIDE              = 11
    INVALID_INPUT_PADDING             = 12
    INVALID_KERNEL_SIZE               = 13
    INVALID_POOLING_SIZE              = 14
    INVALID_POOLING_STRIDE            = 15
    INVALID_ALGORITHM                 = 16
    INVALID_TRANSFORM_STRATEGY        = 17
    UNSUPPORTED_INPUT_SIZE            = 20
    UNSUPPORTED_INPUT_STRIDE          = 21
    UNSUPPORTED_INPUT_PADDING         = 22
    UNSUPPORTED_KERNEL_SIZE           = 23
    UNSUPPORTED_POOLING_SIZE          = 24
    UNSUPPORTED_POOLING_STRIDE        = 25
    UNSUPPORTED_ALGORITHM             = 26
    UNSUPPORTED_TRANSFORM_STRATEGY    = 27
    UNSUPPORTED_ACTIVATION            = 28
    UNSUPPORTED_ACTIVATION_PARAMETERS = 29
    UNINITIALIZED                     = 50
    UNSUPPORTED_HARDWARE              = 51
    OUT_OF_MEMORY                     = 52
    INSUFFICIENT_BUFFER               = 53
    MISALIGNED_BUFFER                 = 54
  end

  enum NNPActivation : LibC::Int
    IDENTITY = 0
    RELU     = 1
  end

  enum NNPConvolutionAlgorithm : LibC::Int
    AUTO          = 0
    FT8X8         = 1
    FT16X16       = 2
    WT8X8         = 3
    IMPLICIT_GEMM = 4
    DIRECT        = 5
    WT8X8_FP16    = 6
  end

  enum NNPConvolutionTransformStrategy : LibC::Int
    COMPUTE    = 1
    PRECOMPUTE = 2
    REUSE      = 3
  end

  struct NNPSize
    width : LibC::SizeT
    height : LibC::SizeT
  end

  struct NNPPadding
    top : LibC::SizeT
    right : LibC::SizeT
    bottom : LibC::SizeT
    left : LibC::SizeT
  end

  struct NNPProfile
    total : LibC::Double
    input_transform : LibC::Double
    kernel_transform : LibC::Double
    output_transform : LibC::Double
    block_multiplication : LibC::Double
  end

  fun initialize = nnp_initialize : NNPStatus
  fun deinitialize = nnp_deinitialize : NNPStatus

  fun nnp_convolution_output = nnp_convolution_output(
    algorithm : NNPConvolutionAlgorithm,
    batch_size : LibC::SizeT,
    input_channels : LibC::SizeT,
    output_channels : LibC::SizeT,
    input_size : NNPSize,
    input_padding : NNPPadding,
    kernel_size : NNPSize,
    input : LibC::Float*,
    kernel : LibC::Float*,
    bias : LibC::Float*,
    output : LibC::Float*,
    workspace_buffer : Void*,
    workspace_size : LibC::SizeT*,
    activation : NNPActivation,
    activation_parameters : Void*,
    threadpool : Void*,
    profile : NNPProfile*
  ) : NNPStatus

  fun nnp_convolution_input_gradient = nnp_convolution_input_gradient(
    algorithm : NNPConvolutionAlgorithm,
    batch_size : LibC::SizeT,
    input_channels : LibC::SizeT,
    output_channels : LibC::SizeT,
    input_size : NNPSize,
    input_padding : NNPPadding,
    kernel_size : NNPSize,
    grad_output : LibC::Float*,
    kernel : LibC::Float*,
    grad_input : LibC::Float*,
    workspace_buffer : Void*,
    workspace_size : LibC::SizeT*,
    activation : NNPActivation,
    activation_parameters : Void*,
    threadpool : Void*,
    profile : NNPProfile*
  ) : NNPStatus

  fun nnp_convolution_kernel_gradient = nnp_convolution_kernel_gradient(
    algorithm : NNPConvolutionAlgorithm,
    batch_size : LibC::SizeT,
    input_channels : LibC::SizeT,
    output_channels : LibC::SizeT,
    input_size : NNPSize,
    input_padding : NNPPadding,
    kernel_size : NNPSize,
    input : LibC::Float*,
    grad_output : LibC::Float*,
    grad_kernel : LibC::Float*,
    workspace_buffer : Void*,
    workspace_size : LibC::SizeT*,
    activation : NNPActivation,
    activation_parameters : Void*,
    threadpool : Void*,
    profile : NNPProfile*
  ) : NNPStatus

  fun nnp_max_pooling_output = nnp_max_pooling_output(
    batch_size : LibC::SizeT,
    channels : LibC::SizeT,
    input_size : NNPSize,
    input_padding : NNPPadding,
    pooling_size : NNPSize,
    pooling_stride : NNPSize,
    input : LibC::Float*,
    output : LibC::Float*
  ) : NNPStatus
end

{% if flag?(:nnpack) %}
  LibNNPACK.initialize
{% end %}
