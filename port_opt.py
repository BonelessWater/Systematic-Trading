import autograd.numpy as np
from autograd import grad

# Tracking error calculation
def tracking_error(curr_pos, ideal_pos):
    error = np.sum((curr_pos - ideal_pos) ** 2)
    return np.sqrt(error / 2)

# Average correlation calculation using NumPy
def average_correlation(positions, correlation_matrix):
    total_correlation = 0
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total_correlation += correlation_matrix[i, j] * positions[i] * positions[j]
            count += 1
    return total_correlation / count if count != 0 else 0

# Gradient of average correlation
def avg_corr_grad(positions, correlation_matrix):
    grad_corr = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                grad_corr[i] += correlation_matrix[i, j] * positions[j]
    return grad_corr / (len(positions) - 1)

# Objective function combining tracking error and average correlation
def objective_function(curr_pos, ideal_pos, correlation_matrix):
    te = tracking_error(curr_pos, ideal_pos)
    avg_corr = average_correlation(curr_pos, correlation_matrix)
    return te + avg_corr

# Gradient descent implementation
def gradient_descent(curr_pos, ideal_pos, correlation_matrix, learning_rate=0.1, iterations=100):
    curr_pos = np.array(curr_pos, dtype=np.float64)
    ideal_pos = np.array(ideal_pos, dtype=np.float64)

    for _ in range(iterations):
        # Tracking error derivative component
        te = tracking_error(curr_pos, ideal_pos)
        if te != 0:
            te_partial_derivative = (curr_pos - ideal_pos) / (2 * te)
        else:
            te_partial_derivative = np.zeros_like(curr_pos)

        # Average correlation derivative component
        corr_partial_derivative = avg_corr_grad(curr_pos, correlation_matrix)

        # Combined derivative
        total_partial_derivative = te_partial_derivative + corr_partial_derivative

        # Update positions
        curr_pos -= learning_rate * total_partial_derivative

        # Convergence check (small change in positions)
        if np.linalg.norm(total_partial_derivative) < 1e-6:
            break

    return curr_pos.tolist()

