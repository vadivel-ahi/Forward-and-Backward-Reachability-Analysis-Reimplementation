"""
Performance Benchmarking
========================
Compare FBRA performance across benchmarks.
Generates table similar to Table 1 in the paper.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
from fbra.verifier import verify_fbra
from fbra.boxes import Box
from experiments.controller import (
    ground_robot_controller,
    BuggyGroundRobotController,
    double_integrator_controller,
    DoubleIntegratorPDController
)
from experiments.dynamics import ground_robot, double_integrator
from experiments.sets import (
    X0_ground_robot, Unsafe_ground_robot,
    X0_double_integrator, Unsafe_double_integrator
)


def benchmark_ground_robot():
    """Benchmark Ground Robot scenarios"""
    
    results = {}
    
    # Safe controller
    start = time.time()
    result_safe = verify_fbra(
        X0_ground_robot, ground_robot_controller, ground_robot,
        Unsafe_ground_robot, T=9, verbose=False
    )
    time_safe = time.time() - start
    
    results['ground_robot_safe'] = {
        'status': result_safe.status,
        'time': time_safe,
        'iterations': result_safe.iterations
    }
    
    # Buggy controller
    buggy = BuggyGroundRobotController()
    start = time.time()
    result_buggy = verify_fbra(
        X0_ground_robot, buggy, ground_robot,
        Unsafe_ground_robot, T=5, verbose=False
    )
    time_buggy = time.time() - start
    
    results['ground_robot_buggy'] = {
        'status': result_buggy.status,
        'time': time_buggy,
        'iterations': result_buggy.iterations
    }
    
    return results


def benchmark_double_integrator():
    """Benchmark Double Integrator scenarios"""
    
    results = {}
    
    # Default controller
    start = time.time()
    result_default = verify_fbra(
        X0_double_integrator, double_integrator_controller, double_integrator,
        Unsafe_double_integrator, T=5, verbose=False
    )
    time_default = time.time() - start
    
    results['double_int_default'] = {
        'status': result_default.status,
        'time': time_default,
        'iterations': result_default.iterations
    }
    
    # PD controller
    pd = DoubleIntegratorPDController()
    start = time.time()
    result_pd = verify_fbra(
        X0_double_integrator, pd, double_integrator,
        Unsafe_double_integrator, T=5, verbose=False
    )
    time_pd = time.time() - start
    
    results['double_int_pd'] = {
        'status': result_pd.status,
        'time': time_pd,
        'iterations': result_pd.iterations
    }
    
    return results


def print_results_table(all_results):
    """Print formatted results table"""
    
    print("\n" + "="*90)
    print("FBRA PERFORMANCE BENCHMARK - Results Table")
    print("="*90)
    
    print(f"\n{'Benchmark':<30} {'Controller':<15} {'Result':<12} {'Time (s)':<12} {'Iterations'}")
    print("-"*90)
    
    benchmark_names = {
        'ground_robot_safe': ('Ground Robot', 'Safe'),
        'ground_robot_buggy': ('Ground Robot', 'Buggy'),
        'double_int_default': ('Double Integrator', 'Default'),
        'double_int_pd': ('Double Integrator', 'PD')
    }
    
    for key, data in all_results.items():
        bench, controller = benchmark_names.get(key, (key, '-'))
        status = data['status']
        time_val = f"{data['time']:.4f}"
        iterations = data['iterations']
        
        # Format status with symbols
        if status == "Safe":
            status_display = f"✓ {status}"
        elif status == "Unsafe":
            status_display = f"✗ {status}"
        else:
            status_display = f"? {status}"
        
        print(f"{bench:<30} {controller:<15} {status_display:<12} {time_val:<12} {iterations}")
    
    print("\n" + "="*90)


def main():
    """Run complete performance benchmark"""
    
    print("\n" + "⚡"*35)
    print("FBRA PERFORMANCE BENCHMARKING")
    print("⚡"*35)
    
    print("\nRunning benchmarks (this may take 1-2 minutes)...")
    
    # Collect all results
    all_results = {}
    
    print("\n1. Ground Robot benchmarks...")
    gr_results = benchmark_ground_robot()
    all_results.update(gr_results)
    
    print("\n2. Double Integrator benchmarks...")
    di_results = benchmark_double_integrator()
    all_results.update(di_results)
    
    # Print table
    print_results_table(all_results)
    
    # Save to file
    with open("results/performance_results.txt", "w") as f:
        f.write("FBRA PERFORMANCE BENCHMARK RESULTS\n")
        f.write("="*90 + "\n\n")
        
        for key, data in all_results.items():
            f.write(f"{key}:\n")
            f.write(f"  Status: {data['status']}\n")
            f.write(f"  Time: {data['time']:.4f}s\n")
            f.write(f"  Iterations: {data['iterations']}\n\n")
    
    print("\n✓ Results saved to: results/performance_results.txt")


if __name__ == "__main__":
    main()