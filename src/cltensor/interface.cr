require "./data_structure"

module NumInternal

  def opencl_get_index_of_element_id : String
    "
    __global static inline int opencl_getIndexOfElementID(
      const int rank,
      __global const int * restrict const shape,
      __global const int * restrict const strides,
      const int offset,
      const int element_id) {
      int real_idx = offset;
      int currentOffset = element_id;
      int dimIdx = 0;
      for (int k = rank - 1; k >= 0; --k) {
        dimIdx = currentOffset % shape[k];
        currentOffset /= shape[k];
        real_idx += dimIdx * strides[k];
      }
      return real_idx;
    }
    "
  end

  def gen_cl_apply3(kern_name : String, ctype : String, op : String) : String
    "
    #{opencl_get_index_of_element_id}

    __kernel void #{kern_name}(const int rank, const int len, __global const int * restrict dst_shape, __global const int * restrict dst_strides, const int dst_offset, __global       #{ctype} * restrict const dst_data, __global const int * restrict A_shape, __global const int * restrict A_strides, const int A_offset, __global const #{ctype} * restrict const A_data, __global const int * restrict B_shape, __global const int * restrict B_strides, const int B_offset, __global const #{ctype} * restrict const B_data)
      {
        // Grid-stride loop
        for (int elemID = get_global_id(0);
        elemID < len;
        elemID += get_global_size(0)) {
          const int dst_real_idx = opencl_getIndexOfElementID(rank, dst_shape, dst_strides, dst_offset, elemID);
          const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
          const int B_real_idx = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
          dst_data[dst_real_idx] = A_data[A_real_idx] #{op} B_data[B_real_idx];
        }
      }
    "
  end

  macro gen_cl_infix_op(dtype, ctype, fn, cname, op)
    def {{fn.id}}(a : ClTensor({{dtype}}), b : ClTensor({{dtype}}))
      result = ClTensor({{dtype}}).new(a.shape)

      cl_kernel = gen_cl_apply3({{cname}}, {{ctype}}, {{op}})
      program = Cl.create_and_build(ClContext.instance.context, cl_kernel, ClContext.instance.device)

      puts cl_kernel

      cl_proc = Cl.create_kernel(program, {{cname}})

      dst = result.layout_on_device
      src_a = a.layout_on_device
      src_b = b.layout_on_device

      # Cl.args(cl_proc, dst.rank, dst.size, dst.shape, dst.strides, 0, dst.data, src_a.shape, src_a.strides, 0, src_a.data, src_b.shape, src_b.strides, 0, src_b.data, dtype: {{dtype}})
    end
  end

  gen_cl_infix_op(Float32, "float", "add", "add_vector", "+")
end

include NumInternal

a = ClTensor(Float32).new([10])
b = ClTensor(Float32).new([10])

puts add(a, b)

a.free
b.free
