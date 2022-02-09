module Num
  # :nodoc:
  # Work arrays pool for lapack routines
  # It isn't thread safe for now because crystal isn't multithreaded
  class WorkPool
    @area = Bytes.new(1024)
    @used = 0

    def get(n) : Bytes
      reallocate(n + @used)
      @area[@used, n].tap { @used += n }
    end

    def get_f32(n) : Slice(Float32)
      get(n*sizeof(Float32)).unsafe_as(Slice(Float32))
    end

    def get_f64(n) : Slice(Float64)
      get(n*sizeof(Float64)).unsafe_as(Slice(Float64))
    end

    def get_cmplx(n) : Slice(LibCblas::ComplexDouble)
      get(n*sizeof(LibCblas::ComplexDouble)).unsafe_as(Slice(LibCblas::ComplexDouble))
    end

    def get_i32(n) : Slice(Int32)
      get(n*sizeof(Int32)).unsafe_as(Slice(Int32))
    end

    def release
      {% if flag?(:release) %}
        @used = 0
      {% else %}
        return if @used == 0
        aused = @used
        @used = 0
        raise "worksize guard failed" unless @area[aused] == 0xDE &&
                                             @area[aused + 1] == 0xAD &&
                                             @area[aused + 2] == 0xBE &&
                                             @area[aused + 3] == 0xEF
      {% end %}
    end

    def reallocate(required_size)
      {% if !flag?(:release) %}
        required_size += 4
      {% end %}
      n = @area.size
      if n < required_size
        while n < required_size
          n = n*2
        end
        @area = Bytes.new(n)
      end
      {% if !flag?(:release) %}
        @area[required_size - 4] = 0xDE
        @area[required_size - 3] = 0xAD
        @area[required_size - 2] = 0xBE
        @area[required_size - 1] = 0xEF
      {% end %}
    end
  end

  # :nodoc:
  WORK_POOL = WorkPool.new
end
