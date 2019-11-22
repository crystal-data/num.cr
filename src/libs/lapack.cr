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

  fun dpotrf = dpotrf_(uplo : Logical*, n : Integer*, a : Double*, lda : Integer*, info : Integer*)
  fun spotrf = spotrf_(uplo : Logical*, n : Integer*, a : Real*, lda : Integer*, info : Integer*)

  fun dpotrs = dpotrs_(uplo : Logical*, n : Integer*, nhrs : Integer*, a : Double*, lda : Integer*, b : Double*, ldb : Integer*, info : Integer*)
  fun spotrs = spotrs_(uplo : Logical*, n : Integer*, nhrs : Integer*, a : Real*, lda : Integer*, b : Real*, ldb : Integer*, info : Integer*)

  fun dgeqrf = dgeqrf_(m : Integer*, n : Integer*, a : Double*, lda : Integer*, tau : Double*, work : Double*, lwork : Integer*, info : Integer*)
  fun sgeqrf = sgeqrf_(m : Integer*, n : Integer*, a : Real*, lda : Integer*, tau : Real*, work : Real*, lwork : Integer*, info : Integer*)

  fun dorgqr = dorgqr_(m : Integer*, n : Integer*, k : Integer*, a : Double*, lda : Integer*, tau : Double*, work : Double*, lwork : Integer*, info : Integer*)
  fun sorgqr = sorgqr_(m : Integer*, n : Integer*, k : Integer*, a : Real*, lda : Integer*, tau : Real*, work : Real*, lwork : Integer*, info : Integer*)

  fun dgesvd = dgesvd_(jobu : Logical*, jobvt : Logical*, m : Integer*, n : Integer*, a : Double*, lda : Integer*, s : Double*, u : Double*, ldu : Integer*, vt : Double*, ldvt : Integer*, work : Double*, lwork : Integer*, info : Integer*)
  fun sgesvd = sgesvd_(jobu : Logical*, jobvt : Logical*, m : Integer*, n : Integer*, a : Real*, lda : Integer*, s : Real*, u : Real*, ldu : Integer*, vt : Real*, ldvt : Integer*, work : Real*, lwork : Integer*, info : Integer*)

  fun dsyev = dsyev_(jobz : Logical*, uplo : Logical*, n : Integer*, a : Double*, lda : Integer*, w : Double*, work : Double*, lwork : Integer*, info : Integer*)
  fun ssyev = ssyev_(jobz : Logical*, uplo : Logical*, n : Integer*, a : Real*, lda : Integer*, w : Real*, work : Real*, lwork : Integer*, info : Integer*)

  fun dgeev = dgeev_(jobvl : Logical*, jobvr : Logical*, n : Integer*, a : Double*, lda : Integer*, wr : Double*, wi : Double*, vl : Double*, ldvl : Integer*, vr : Double*, ldvr : Integer*, work : Double*, lwork : Integer*, info : Integer*)
  fun sgeev = sgeev_(jobvl : Logical*, jobvr : Logical*, n : Integer*, a : Real*, lda : Integer*, wr : Real*, wi : Real*, vl : Real*, ldvl : Integer*, vr : Real*, ldvr : Integer*, work : Real*, lwork : Integer*, info : Integer*)

  fun dlange = dlange_(norm : Logical*, m : Integer*, n : Integer*, a : Double*, lda : Integer*, work : Double*) : Double
  fun slange = slange_(norm : Logical*, m : Integer*, n : Integer*, a : Real*, lda : Integer*, work : Real*) : Real

  fun dgecon = dgecon_(norm : Logical*, n : Integer*, a : Double*, lda : Integer*, anorm : Double*, rcond : Double*, work : Double*, iwork : Integer*, info : Integer*)
end
