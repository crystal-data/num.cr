require "./dtype"

@[Link("lapack")]
lib LibLapack
  alias Integer = LibC::Int
  alias Real = LibC::Float
  alias Double = LibC::Double
  alias Logical = LibC::Char
  alias Ftnlen = LibC::Int
  alias LFp = Pointer(Void)
  alias UInt = LibC::SizeT
  alias Indexer = UInt64 | Int32
  alias BNum = Int32 | Float64 | Float32

  enum LapackTranspose
    Trans = 54
  end

  fun dgetrf = dgetrf_(m : Integer*, n : Integer*, a : Double*, lda : Integer*, ipiv : Integer*, info : Integer*)
  fun sgetrf = sgetrf_(m : Integer*, n : Integer*, a : Real*, lda : Integer*, ipiv : Integer*, info : Integer*) : Integer
  fun dgetri = dgetri_(n : Integer*, a : Double*, lda : Integer*, ipiv : Integer*,
                       work : Double*, lwork : Integer*, info : Integer*) : Integer
  fun sgetri = sgetri_(n : Integer*, a : Real*, lda : Integer*, ipiv : Integer*,
                       work : Real*, lwork : Integer*, info : Integer*) : Integer
end
