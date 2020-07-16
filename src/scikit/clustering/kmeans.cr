# Copyright (c) 2020 Crystal Data Contributors
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

require "../../libs/local"

module SciKit
  # In the K-Means problem, a set of N points X(I) in M-dimensions is given.
  # The goal is to arrange these points into K clusters, with each cluster
  # having a representative point Z(J), usually chosen as the centroid
  # of the points in the cluster. The energy of each cluster is
  #
  #         E(J) = Sum ( all points X(I) in cluster J ) || X(I) - Z(J) ||^2
  #
  # For a given set of clusters, the total energy is then simply the
  # sum of the cluster energies E(J). The goal is to choose the clusters
  # in such a way that the total energy is minimized. Usually, a point X(I)
  # goes into the cluster with the closest representative point Z(J). So
  # to define the clusters, it's enough simply to specify the locations
  # of the cluster representatives.
  #
  # This is actually a fairly hard problem. Most algorithms do reasonably
  # well, but cannot guarantee that the best solution has been found.
  # It is very common for algorithms to get stuck at a solution which
  # is merely a "local minimum". For such a local minimum, every slight
  # rearrangement of the solution makes the energy go up; however a major
  # rearrangement would result in a big drop in energy.
  #
  # A simple algorithm for the problem is known as "H-Means". It alternates
  # between two procedures:
  #
  # Using the given cluster centers, assign each point to the cluster
  # with the nearest center;
  # Using the given cluster assignments, replace each cluster center
  # by the centroid or average of the points in the cluster.
  # These steps are repeated until no points are moved, or some other
  # termination criterion is reached.
  # A more sophisticated algorithm, known as "K-Means", takes advantage
  # of the fact that it is possible to quickly determine the decrease in
  # energy caused by moving a point from its current cluster to another.
  # It repeats the following procedure:
  #
  # For each point, move it to another cluster if that would lower the energy.
  # If you move a point, immediately update the cluster centers of the two
  # affected clusters.
  # This procedure is repeated until no points are moved, or some other
  # termination criterion is reached.
  #
  # References
  # ----------
  # John Hartigan, Manchek Wong,
  # Algorithm AS 136: A K-Means Clustering Algorithm,
  # Applied Statistics,
  # Volume 28, Number 1, 1979, pages 100-108.
  # Wendy Martinez, Angel Martinez,
  # Computational Statistics Handbook with MATLAB,
  # pages 373-376,
  # Chapman and Hall / CRC, 2002.
  # David Sparks,
  # Algorithm AS 58: Euclidean Cluster Analysis,
  # Applied Statistics,
  # Volume 22, Number 1, 1973, pages 126-130.
  def kmeans(
    obs : Tensor(Float64),
    guess : Tensor(Float64),
    iter : Int = 20
  )
    if !obs.flags.contiguous?
      obs = obs.dup(Num::RowMajor)
    end
    guess = guess.dup(Num::RowMajor)
    m, n = obs.shape
    k, _ = guess.shape
    ic = Tensor(Int32).new([m])
    nc = Tensor(Int32).new([k])
    ws = Tensor(Float64).new([k])
    LibKmeans.kmns(
      obs.to_unsafe,
      m,
      n,
      guess.to_unsafe,
      k,
      ic.to_unsafe,
      nc.to_unsafe,
      iter,
      ws.to_unsafe,
      out ifault
    )
    {guess, Num.mean(ws)}
  end
end
