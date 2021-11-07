# Copyright (c) 2021 Crystal Data Contributors
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

VALID_CHARS    = "abcdefghijklmopqABC".chars
SIZES          = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4]
DIMENSION_DICT = Hash.zip(VALID_CHARS, SIZES)

def build_views(einsum_string : String) : Array(Tensor(Float64, CPU(Float64)))
  views = [] of Tensor(Float64, CPU(Float64))
  terms = einsum_string.split("->")[0].split(",")
  terms.each do |term|
    dims = term.chars.map { |c| DIMENSION_DICT[c] }
    views << Tensor(Float64, CPU(Float64)).ones(dims)
  end
  views
end

def correct_output_size(einsum_string : String, operands : Array(Tensor(Float64, CPU(Float64))))
  sc = Num::Einsum::SizedContraction.new(einsum_string, operands)
  full_einsum_string = sc.as_einsum_string
  terms = full_einsum_string.split("->")[1].chars
  terms.map { |c| DIMENSION_DICT[c] }
end

macro build_test(cases)
  {% for test_case in cases %}
    it "Contracts operands into the correct shape for #{{{test_case}}}" do
      views = build_views({{ test_case }})
      output_size = correct_output_size({{ test_case }}, views)

      result = Num::Einsum.einsum({{ test_case }}, views)

      result.shape.should eq output_size
    end
  {% end %}
end

describe Num::Einsum do
  build_test [
    # Test hadamard-like products
    "a,ab,abc->abc",
    "a,b,ab->ab",

    # Test index-transformations
    "ea,fb,gc,hd,abcd->efgh",
    "ea,fb,abcd,gc,hd->efgh",
    "abcd,ea,fb,gc,hd->efgh",

    # # Test complex contractions
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "cd,bdhe,aidb,hgca,gc,hgibcd,hgac",
    "abhe,hidj,jgba,hiab,gab",
    "bde,cdh,agdb,hica,ibd,hgicd,hiac",
    "chd,bde,agbc,hiad,hgc,hgi,hiad",
    "chd,bde,agbc,hiad,bdi,cgh,agdb",
    "bdhe,acad,hiab,agac,hibd",

    # Test collapse
    "ab,ab,c->c",
    "ab,ab,cd,cd->ac",
    "ab,ab,cd,cd->cd",

    # Test outer prodcuts
    "ab,cd,ef->abcdef",
    "ab,cd,ef->acdf",
    "ab,cd,de->abcde",
    "ab,cd,de->be",
    "ab,bcd,cd->abcd",
    "ab,bcd,cd->abd",
    #
    # # Random test cases that have previously failed
    "eb,cb,fb->cef",
    "dd,fb,be,cdb->cef",
    "bca,cdb,dbf,afc->",
    "dcc,fce,ea,dbf->ab",
    "fdf,cdd,ccd,afe->ae",
    "abcd,ad",
    "ed,fcd,ff,bcf->be",
    "baa,dcf,af,cde->be",
    "bd,db,eac->ace",
    "fff,fae,bef,def->abd",
    "efc,dbc,acf,fd->abe",
    #
    # # Inner products
    "ab,ab",
    "ab,ba",
    "abc,abc",
    "abc,bac",
    "abc,cba",
    #
    # # GEMM test cases
    "ab,bc",
    "ab,cb",
    "ba,bc",
    "ba,cb",
    "abcd,cd",
    "abcd,ab",
    "abcd,cdef",
    "abcd,cdef->feba",
    "abcd,efdc",
    #
    # # Inner than dot
    "aab,bc->ac",
    "ab,bcc->ac",
    "aab,bcc->ac",
    "baa,bcc->ac",
    "aab,ccb->ac",
    #
    # # Randomly build test caes
    "aab,fa,df,ecc->bde",
    "ecb,fef,bad,ed->ac",
    "bb,ff,be->e",
    "afd,ba,cc,dc->bf",
    "adb,bc,fa,cfc->d",
    "bbd,bda,fc,db->acf",
    "dba,ead,cad->bce",
    "aef,fbc,dca->bde",
  ]
end
