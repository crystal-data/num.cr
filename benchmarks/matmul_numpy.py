import numpy as np

def main(n):
    print(f"\nNumpy matmul {n}x{n}")
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.dot(a, b)
    print(c[n//2, n//2])

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main(100)
