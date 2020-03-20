require "json"

module Num::InputOutput
  TYPE_HASH = {
    UInt8   => "|u1",
    UInt16  => "<u2",
    UInt32  => "<u4",
    Int8    => "|i1",
    Int16   => "<i2",
    Int32   => "<i4",
    Float32 => "<f4",
    Float64 => "<f8",
  }

  UNTYPE_HASH = {
    "|u1" => UInt8,
    "<u2" => UInt16,
    "<u4" => UInt32,
    "|i1" => Int8,
    "<i2" => Int16,
    "<i4" => Int32,
    "<f4" => Float32,
    "<f8" => Float64,
  }

  def save(filename, tensor)
    t = tensor.dup
    data = t.buffer.unsafe_as(Pointer(UInt8)).to_slice(t.bytesize * t.size)
    header = "\x93NUMPY"
    version = "\x01\x00"

    outf = File.open(filename, "w")
    outf << header
    outf << version

    dt = TYPE_HASH[t.dtype]

    shape = "#{t.shape}"[1...-1]
    if t.ndims == 1
      shape += ','
    end

    metadata = "{'descr': '#{dt}', 'fortran_order': False, 'shape': (#{shape}), }\n"
    headersize = UInt16.new(metadata.bytesize)
    hptr = Pointer(UInt16).malloc(1) { |_| headersize }
    hptr = hptr.unsafe_as(Pointer(UInt8)).to_slice(2)

    outf.write(hptr)
    outf << metadata
    outf.write(data)
    outf.close
  end

  def totalsize(shape)
    shape.reduce { |i, j| i * j }
  end

  macro read_cast_return(dtype)
    newdata = Bytes.new(sizeof({{dtype}}) * newsize)
    inf.read_fully(newdata)
    ptr = newdata.to_unsafe.unsafe_as(Pointer({{dtype}}))
    Tensor(U).new(newshape) { |i| U.new(ptr[i]) }
  end

  def load(filename, dtype : U.class = Float64) forall U
    inf = File.open(filename, "r")
    toss = Bytes.new(8)
    header = Bytes.new(2)
    inf.read_fully(toss)
    inf.read_fully(header)
    headersize = IO::Memory.new(header).read_bytes(Int16, IO::ByteFormat::LittleEndian)
    header = inf.read_string(headersize).downcase.tr("()'", "[]\"").gsub(/,]|],/, "]")
    jhead = JSON.parse(header)

    desc = jhead["descr"]
    shape = jhead["shape"]
    newshape = [0] * shape.size
    newshape = newshape.map_with_index { |_, i| shape[i].as_i }

    newsize = totalsize(newshape)
    case desc
    when "|u1"
      read_cast_return UInt8
    when "<u2"
      read_cast_return UInt16
    when "<u4"
      read_cast_return UInt32
    when "|i1"
      read_cast_return Int8
    when "<i2"
      read_cast_return Int16
    when "<i4"
      read_cast_return Int32
    when "<f4"
      read_cast_return Float32
    when "<f8"
      read_cast_return Float64
    else
      raise "Dtype #{desc} is not currently supported by Num.cr"
    end
  end
end
