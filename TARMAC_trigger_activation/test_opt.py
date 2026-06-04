from tarmac_optimized import tarmac_test_generation
res = tarmac_test_generation("benchmarks/TRIT-TC/c2670_T000/c2670_T000.v", "c2670_T000", num_vectors=1, rare_threshold=0.1)
print(res)
