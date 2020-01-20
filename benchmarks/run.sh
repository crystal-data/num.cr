ruby xtime.rb ./matmul_num 5000
ruby xtime.rb python matmul_numpy.py 5000
ruby xtime.rb ./matmul_arraymancer 5000

ruby xtime.rb ./elementwise_num 100000000
ruby xtime.rb python elementwise_numpy.py 100000000
ruby xtime.rb ./elementwise_arraymancer 100000000
