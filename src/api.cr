# require "./libs/cblas"
# require "./libs/lapack"
# require "./libs/clblast"
#
require "./tensor/internal/shape"
require "./tensor/internal/broadcast"
# require "./tensor/internal/print"
#
require "./tensor/enums"
require "./tensor/data_structure"
require "./tensor/allocation"
require "./tensor/convert"
# require "./tensor/iteration"
require "./tensor/creation"
# require "./tensor/index"
# require "./tensor/shifting"
# require "./tensor/linalg"

require "./tensor/backends2/storage"
require "./tensor/backends2/hostptr/yield_iterators"
#
# require "./tensor/backends/agnostic/storage"
# require "./tensor/backends/linalg/linalg_has_hostptr"
# require "./tensor/backends/linalg/work_has_hostptr"
# require "./tensor/backends/linalg/definitions_has_hostptr"
#
require "./tensor/backends2/cpu/storage"
# require "./tensor/backends/cpu/convert"
# require "./tensor/backends/cpu/allocation"
# require "./tensor/backends/cpu/private/yield_iterators"
# require "./tensor/backends/cpu/iteration"
# require "./tensor/backends/cpu/index"
# require "./tensor/backends/cpu/linalg"
#
require "./tensor/backends2/opencl/private/global_state"
require "./tensor/backends2/opencl/storage"
require "./tensor/backends2/opencl/convert"
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
