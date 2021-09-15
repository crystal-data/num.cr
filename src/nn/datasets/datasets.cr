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

require "http"
require "digest"

module Num::NN
  extend self

  BASE_DATASET_CACHE_PATH = "#{Path.home}/.cache/num.cr/datasets"

  private def load_dataset_http(url : String)
    Dir.mkdir_p(BASE_DATASET_CACHE_PATH) unless Dir.exists?(BASE_DATASET_CACHE_PATH)

    cache_file_name = Digest::SHA1.digest(url).to_slice.hexstring
    cache_file_path = "#{BASE_DATASET_CACHE_PATH}/#{cache_file_name}"

    return File.read(cache_file_path) if File.exists?(cache_file_path)

    response = HTTP::Client.get(url)
    content = response.body

    File.write(cache_file_path, content)

    content
  end
end
