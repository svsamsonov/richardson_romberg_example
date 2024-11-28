import numpy as np
from multiprocessing import Pool

# Define the function and its derivative
def f(x):
    return x**2 + np.cos(x)

def grad_f(x):
    return 2 * x - np.sin(x)

def stochastic_gradient_descent_online(start_point, gamma=0.1, iterations=1000, seed=42):
    """
    Perform gradient descent with noisy gradients in an online setting.
    Generate new noisy gradients on each iteration.

    Parameters:
    - fixed_point: Point at which the gradient is evaluated.
    - gamma: Base step size.
    - iterations: Number of iterations.
    - seed: Random seed for reproducibility.

    Returns:
    - Averaged parameters and gradients over the second half of the trajectory.
    """
    np.random.seed(seed)  # Set random seed for noise
    x_gamma = start_point
    x_2gamma = start_point

    avg_x_gamma = 0.0
    avg_x_2gamma = 0.0
    grad_fixed_point_avg = 0.0
    count_avg = 0

    for it in range(iterations):
        # Compute noisy gradient
        true_grad = grad_f(x_gamma)
        noise = np.random.normal(0, 1)  # Standard normal noise
        noisy_grad = true_grad + noise

        # Update with step size gamma
        x_gamma -= gamma * noisy_grad

        # Update with step size 2*gamma
        true_grad_2gamma = grad_f(x_2gamma)
        x_2gamma -= 2 * gamma * (true_grad_2gamma + noise)


        # Compute gradient at fixed point (no noise added here)
        grad_fixed_point = noise

        # Update averages for the second half of the trajectory
        if it >= iterations // 2:
            count_avg += 1
            avg_x_gamma += (x_gamma - avg_x_gamma) / count_avg
            avg_x_2gamma += (x_2gamma - avg_x_2gamma) / count_avg
            grad_fixed_point_avg += (grad_fixed_point - grad_fixed_point_avg) / count_avg

    return avg_x_gamma, avg_x_2gamma, grad_fixed_point_avg

def run_gradient_descent(args):
    """
    Wrapper to run stochastic gradient descent for all iterations in iterations_list.
    Parameters:
    - args: Tuple containing fixed_point, gamma, iterations_list, and seed.

    Returns:
    - Averaged results for all iterations.
    """
    start_point, iterations_list, seed = args
    results = []
    for iterations in iterations_list:
        gamma = 1./np.sqrt(iterations)
        avg_x_gamma, avg_x_2gamma, grad_fixed_point_avg = stochastic_gradient_descent_online(
            start_point, gamma, iterations, seed
        )
        results.append(
            np.array([avg_x_gamma,avg_x_2gamma,grad_fixed_point_avg])
        )
    return results

def process_in_blocks(args, nbcores, n_blocks):
    """
    Process tasks in sequential blocks of parallel processes.

    Parameters:
    - args: List of arguments for the worker function.
    - nbcores: Number of processes to run in parallel per block.
    - n_blocks: Total number of blocks.
    """
    total_tasks = len(args)
    tasks_per_block = total_tasks // n_blocks

    all_results = []
    for block_idx in range(n_blocks):
        print(f"Processing block {block_idx + 1} of {n_blocks}...")
        start_idx = block_idx * tasks_per_block
        end_idx = start_idx + tasks_per_block
        block_args = args[start_idx:end_idx]

        # Use Pool to process the current block
        with Pool(processes=nbcores) as pool:
            block_results = pool.map(run_gradient_descent, block_args)

        all_results.extend(block_results)

    return all_results

if __name__ == "__main__":
    # Parameters
    start_point = 5.0 # Starting point
    n_powers = 14
    iterations_list = [500 * (2**i) for i in range(n_powers)]
    total_threads = 320
    nbcores = 8  # Number of cores to use, adjust to your computer
    n_blocks = total_threads // nbcores

    # Random seeds for each thread
    random_seeds = [42 + i for i in range(total_threads)]

    # Combine parameters into a single iterable for parallel execution
    args = [(start_point, iterations_list, seed) for seed in random_seeds]

    # Run SGD in parallel
    parallel_results = process_in_blocks(args, nbcores=nbcores, n_blocks=n_blocks)
    parallel_results = np.asarray(parallel_results)
    print(parallel_results.shape)
    # Save results
    np.save("stochastic_gradient_descent_results.npy", parallel_results)
    print("Results saved to stochastic_gradient_descent_results.npy")
