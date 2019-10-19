require "../libs/dtype"
require "../libs/gsl"
require "../vector/*"
require "../core/exceptions"

macro assignment_helper(dtype, matrix_prefix, vector_prefix, matrix_type, vector_type)
  module LL
    extend self

    # Normalizes a slice range to allow indexing to work with gsl.  This means
    # bounding nil ranges and disallowing inclusive ranges
    def convert_range_to_slice(range : Range(Int32 | Nil, Int32 | Nil), size : UInt64)
      if !range.excludes_end?
        raise Bottle::Core::Exceptions::RangeError.new("Vectors do not support indexing with inclusive ranges. Always use '...'")
      end

      i = range.begin
      j = range.end

      start = (i.nil? ? 0 : i).as(UInt64 | Int32).to_u64
      finish = (j.nil? ? size : j).as(UInt64 | Int32).to_u64

      return start...finish
    end

    def get(v : Pointer({{vector_type}}), i : Indexer)
      LibGsl.{{vector_prefix}}_get(v, i)
    end

    def get(m : Pointer({{matrix_type}}), i : Indexer, j : Indexer)
      LibGsl.{{matrix_prefix}}_get(m, i, j)
    end

    def slice(m : Vector({{vector_type}}, {{dtype}}), range)
      rng = convert_range_to_slice(range, m.size)
      subv = LibGsl.{{vector_prefix}}_subvector(m.ptr, rng.begin, rng.end - rng.begin)
      return Vector.new subv.vector, subv.vector.data
    end

    def set(v : Pointer({{vector_type}}), i : Indexer, x : BNum)
      LibGsl.{{vector_prefix}}_set(v, i, x)
    end

    def set(m : Pointer({{matrix_type}}), i : Indexer, j : Indexer, x : BNum)
      LibGsl.{{matrix_prefix}}_set(m, i, j, x)
    end

    def full(v : Pointer({{vector_type}}), x : BNum)
      LibGsl.{{vector_prefix}}_set_all(v, x)
    end

    def full(m : Pointer({{matrix_type}}), x : BNum)
      LibGsl.{{matrix_prefix}}_set_all(m, x)
    end

    def zero_out(v : Pointer({{vector_type}}))
      LibGsl.{{vector_prefix}}_set_zero(v)
    end

    def zero_out(m : Pointer({{matrix_type}}))
      LibGsl.{{matrix_prefix}}_set_zero(m)
    end

    def set_basis(v : Pointer({{vector_type}}), i : Indexer)
      LibGsl.{{vector_prefix}}_set_basis(v, i)
    end
  end
end

assignment_helper Float64, gsl_matrix, gsl_vector, LibGsl::GslMatrix, LibGsl::GslVector
assignment_helper Float32, gsl_matrix_float, gsl_vector_float, LibGsl::GslMatrixFloat, LibGsl::GslVectorFloat
