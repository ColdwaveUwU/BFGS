import numpy as np
import time

def gradient_descent(func, grad_func, x0, learning_rate=0.01, max_iter=100, tol=1e-5):
    x = x0
    for i in range(max_iter):
        grad = grad_func(x)
        grad = np.clip(grad, -1e5, 1e5)
        x_new = x - learning_rate * grad
        if np.linalg.norm(grad) < tol:
            print(f"Gradient Descent converged at iteration {i+1}")
            return x_new
        x = x_new
    
    print(f"Gradient Descent: Max iterations reached at iteration {max_iter}.")
    return x

def func(x):
    return (x - 5)**2 + 3 * (x - 5)**4

def grad_func(x):
    return 2 * (x - 5) + 12 * (x - 5)**3

def bfgs(func, grad_func, x0, max_iter=100, tol=1e-5):
    x = x0
    n = len(x0)
    H = np.eye(n)
    grad = grad_func(x)

    for i in range(max_iter):
        p = -np.dot(H, grad)
        alpha = 1.0
        x_new = x + alpha * p
        grad_new = grad_func(x_new)
        if np.linalg.norm(grad_new) < tol:
            print(f"BFGS converged at iteration {i+1}")
            return x_new
        s = x_new - x
        y = grad_new - grad
        rho = 1.0 / np.dot(y, s)
        H = (np.eye(n) - rho * np.outer(s, y)) @ H @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
        grad = grad_new

    print(f"BFGS: Max iterations reached at iteration {max_iter}.")
    return x

def compare_methods():
    x0 = np.array([6.0])
    print("Running Gradient Descent...\n")
    
    start_time = time.time()
    solution_gd = gradient_descent(func, grad_func, x0, learning_rate=0.01, max_iter=100)
    gd_time = time.time() - start_time
    print(f"Gradient Descent solution: {solution_gd}")
    print(f"Gradient Descent took {gd_time:.4f} seconds\n")
    
    print("Running BFGS...\n")
    
    start_time = time.time()
    solution_bfgs = bfgs(func, grad_func, x0, max_iter=100)
    bfgs_time = time.time() - start_time
    print(f"BFGS solution: {solution_bfgs}")
    print(f"BFGS took {bfgs_time:.4f} seconds\n")
    
    print(f"Gradient Descent took {gd_time:.4f} seconds with {solution_gd} as solution.")
    print(f"BFGS took {bfgs_time:.4f} seconds with {solution_bfgs} as solution.")

compare_methods()
