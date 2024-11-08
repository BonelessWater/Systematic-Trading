import autograd.numpy as np
from autograd import grad

# Define the correlation matrix
correlation_matrix = np.array([
    [1, 0.5, 0.3, 0.2, 0.1],
    [0.5, 1, 0.4, 0.3, 0.2],
    [0.3, 0.4, 1, 0.6, 0.3],
    [0.2, 0.3, 0.6, 1, 0.4],
    [0.1, 0.2, 0.3, 0.4, 1]
])

# Define functions
def tracking_error(curr_pos, ideal_pos):
    error = np.sum((curr_pos - ideal_pos) ** 2)
    return np.sqrt(error / 2)

def average_correlation(positions):
    total_correlation = 0
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total_correlation += correlation_matrix[i][j] * positions[i] * positions[j]
            count += 1
    return total_correlation / count if count != 0 else 0

avg_corr_grad = grad(average_correlation)

def objective_function(curr_pos, ideal_pos):
    te = tracking_error(curr_pos, ideal_pos)
    avg_corr = average_correlation(curr_pos)
    return te + avg_corr

def adam_optimization(curr_pos, ideal_pos, learning_rate=1, iterations=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
    curr_pos = np.array(curr_pos, dtype=np.float64)
    ideal_pos = np.array(ideal_pos, dtype=np.float64)

    # Initialize moment estimates
    m = np.zeros_like(curr_pos)  # First moment vector (for momentum)
    v = np.zeros_like(curr_pos)  # Second moment vector (for adaptive learning rate)
    t = 0  # Timestep

    for _ in range(iterations):
        t += 1

        # Tracking error derivative component
        te = tracking_error(curr_pos, ideal_pos)
        if te != 0:
            te_partial_derivative = (curr_pos - ideal_pos) / (2 * te)
        else:
            te_partial_derivative = np.zeros_like(curr_pos)

        # Average correlation derivative component using autograd
        corr_partial_derivative = avg_corr_grad(curr_pos)

        # Combined derivative
        total_partial_derivative = te_partial_derivative + 0.005*corr_partial_derivative

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * total_partial_derivative
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (total_partial_derivative ** 2)

        # Compute bias-corrected first and second moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update positions using Adam
        curr_pos -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Calculate the new objective function value
        obj_value = objective_function(curr_pos, ideal_pos)
        if obj_value < 1e-80:
            break
    
    # Round positions to integer values only once at the end
    return np.round(curr_pos).astype(int).tolist()

# Initial positions
ideal_pos = [15.3, 21.3, 11.7, 7.4, 19.2]
curr_pos = [0, 0, 0, 0, 0]

print(f'Starting positions {curr_pos}')
print(f'Starting objective function value {objective_function(np.array(curr_pos), np.array(ideal_pos))}')

# Run Adam optimization
curr_pos = adam_optimization(curr_pos, ideal_pos)

print(f'Ending positions {curr_pos}')
print(f'Ending objective function value {objective_function(np.array(curr_pos), np.array(ideal_pos))}')
