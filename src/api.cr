# require "./libs/cblas"
# require "./libs/lapack"
# require "./libs/clblast"

require "complex"

require "./tensor/internal/shape"
require "./tensor/internal/broadcast"
require "./tensor/internal/print"
require "./tensor/internal/enums"

require "./tensor/data_structure"
require "./tensor/allocation"
require "./tensor/iteration"
require "./tensor/index"
require "./tensor/manipulate"
require "./tensor/math"
require "./tensor/convert"

require "./tensor/backends/util_storage"
require "./tensor/backends/util_manipulate"
require "./tensor/backends/util_index"

require "./tensor/backends/hostptr/unsafe_iter"
require "./tensor/backends/hostptr/yield_iterators"
#
# require "./tensor/backends/agnostic/storage"
# require "./tensor/backends/linalg/linalg_has_hostptr"
# require "./tensor/backends/linalg/work_has_hostptr"
# require "./tensor/backends/linalg/definitions_has_hostptr"
#
require "./tensor/backends/cpu/impl_allocation"
require "./tensor/backends/cpu/impl_manipulate"
require "./tensor/backends/cpu/impl_iteration"
require "./tensor/backends/cpu/impl_index"
require "./tensor/backends/cpu/impl_math"
require "./tensor/backends/cpu/impl_data_structure"
# require "./tensor/backends/cpu/convert"
# require "./tensor/backends/cpu/allocation"
# require "./tensor/backends/cpu/private/yield_iterators"
# require "./tensor/backends/cpu/linalg"
#
require "./tensor/backends/opencl/private/global_state"
require "./tensor/backends/opencl/impl_allocation"
require "./tensor/backends/opencl/impl_data_structure"
require "./tensor/backends/opencl/impl_convert"
require "./tensor/backends/opencl/impl_math"
require "./tensor/backends/opencl/impl_index"
require "./tensor/backends/opencl/impl_manipulate"
# require "./tensor/backends/opencl/allocation"
# require "./tensor/backends/opencl/linalg"
#
# {% if flag?(:arrow) %}
#   require "./tensor/backends/arrow/storage"
#   require "./tensor/backends/arrow/index"
#   require "./tensor/backends/arrow/allocation"
#   require "./tensor/backends/arrow/convert"
#   require "./tensor/backends/arrow/iteration"
#   require "./tensor/backends/arrow/private/gobject"
#   require "./tensor/backends/arrow/arrow_primitives"
# {% end %}
#
# require "./extensions/array"
