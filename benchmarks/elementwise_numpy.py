import numpy as np

def main(n):
    print(f"\nNumpy elementwise {n} 1 + i / 1 + j / k")
    i = np.random.rand(n)
    j = np.random.rand(n)
    k = np.random.rand(n)
    res = 1 + i / 1 + j * k
    print(res[n//2])

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main(100)
