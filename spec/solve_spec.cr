require "./spec_helper"

describe "Solve" do
  it "Solves a system of linear equations" do
    a = [[3.0, 1.0], [1.0, 2.0]].to_tensor

    b = [9.0, 8.0].to_tensor

    a.solve(b).should eq [2, 3]
  end
end