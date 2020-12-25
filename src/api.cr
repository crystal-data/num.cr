require "./tensor/internal/shape"
require "./tensor/internal/broadcast"

require "./tensor/enums"
require "./tensor/data_structure"
require "./tensor/allocation"
require "./tensor/convert"
require "./tensor/iteration"
require "./tensor/creation"

require "./tensor/backends/agnostic/storage"

require "./tensor/backends/cpu/storage"
require "./tensor/backends/cpu/convert"
require "./tensor/backends/cpu/allocation"
require "./tensor/backends/cpu/private/yield_iterators"
require "./tensor/backends/cpu/iteration"

require "./tensor/backends/opencl/private/global_state"
require "./tensor/backends/opencl/storage"
require "./tensor/backends/opencl/convert"
require "./tensor/backends/opencl/allocation"

{% if flag?(:arrow) %}
  require "./tensor/backends/arrow/storage"
  require "./tensor/backends/arrow/allocation"
  require "./tensor/backends/arrow/convert"
  require "./tensor/backends/arrow/iteration"
  require "./tensor/backends/arrow/private/gobject"
  require "./tensor/backends/arrow/arrow_primitives"
{% end %}
