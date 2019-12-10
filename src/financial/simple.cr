require "../tensor/tensor"
require "../core/math"

# Credit to https://github.com/drum445 for implementing most of these, they
# worked right out of the game with vectorized methods.

module Num::Financial
  def fv(rate, nper, pmt, pv, pay_when = 0)
    if rate == 0
      -1 * (pv + pmt * nper)
    else
      pow = (1 + rate) ** nper
      (pmt * (1 + rate * pay_when) * (1 - pow) / rate) - pv * pow
    end
  end

  private def pv_f(rate, nper)
    (1 + rate) ** nper
  end

  private def pv_annuity(rate, nper, pmt, fv, pay_when)
    (fv + pmt * (1 + rate * pay_when) * (((1 + rate) ** nper) - 1) / rate)
  end

  def pv(rate, nper, pmt, fv = 0, pay_when = 0)
    if rate == 0
      -pmt * nper - fv
    else
      annuity = pv_annuity(rate, nper, pmt, fv, pay_when)
      fv = pv_f(rate, nper)
      -1 * (annuity / fv)
    end
  end

  def pmt(rate, nper, pv, fv, pay_when = 0)
    pmt = 0
    pv = pv * -1
    pv_minus_fv = pv - fv

    if rate == 0
      pv_minus_fv / nper
    else
      rate_to_nper = (rate + 1) ** nper
      pmt = (pv_minus_fv * (rate * rate_to_nper)) / (rate_to_nper - 1)

      fv_rate = fv * rate
      pmt = (pmt + fv_rate)

      if pay_when == 1
        pmt = pmt / (1 + rate)
      end
      pmt
    end
  end

  def nper(rate, pmt, pv, fv, pay_when = 0)
    if rate == 0
      -(pv + fv) / pmt
    else
      num = pmt * (1 + rate * pay_when) - fv * rate
      den = pv * rate + pmt * (1 + rate * pay_when)
      BMath.log10(num / den) / BMath.log10(1 + rate)
    end
  end
end
