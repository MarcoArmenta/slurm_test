import numpy as np
import argparse
from multiprocessing import Pool

# This is the BIG part of the code that comes from somewhere else
def generate_random_matrix(rows, cols, n):
    print("Genearate matrix: ", n)
    return np.random.rand(rows, cols)

def multiply_matrices(matrix1, matrix2, n):
    print("Matrix number: ", n)
    return np.dot(matrix1, matrix2)

# This part is the script that you build to run on the cluster
def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--array_index",
        default=10,
        type=int,
        help="The index for default experiment",
    )
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=8,
        help="Number of cpu cores for parallel computation",
    )
    parser.add_argument(
        "--num_matrices",
        type=int,
        default=100,
        help="Number of matrices",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    array_index = args.array_index
    cpu_cores = args.cpu_cores
    num_matrices = args.num_matrices

    rows = array_index
    cols = array_index

    parameters_matrices = [(rows, cols, n) for n in range(num_matrices)]

    with Pool(processes=cpu_cores) as pool:
        matrices_1 = pool.starmap(generate_random_matrix, parameters_matrices)
        matrices_2 = pool.starmap(generate_random_matrix, parameters_matrices)

    parameters = [(matrices_1[i], matrices_2[j], len(matrices_1)*j+i) for i in range(len(matrices_1)) for j in range(len(matrices_2))] # 100*100 = 10,000

    with Pool(processes=cpu_cores) as pool:
        results = pool.starmap(multiply_matrices, parameters)

    np.save("results.npy", results)


if __name__ == "__main__":
    main()
