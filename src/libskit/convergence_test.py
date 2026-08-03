#!/usr/bin/env python3
#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Convergence Test Module for LIBS Gradient Descent Optimization

Simple convergence testing with parameter ranges and individual run pickle files.
"""

import numpy as np
import random
import multiprocessing as mp
from functools import partial
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ._libskit import Experiment, LTEModel, GenericSpectrometer, Spectrum, LossType
from .gradient_descent import custom_gradient_descent


def run_single_convergence_test(run_index: int, experiment_data: Dict, parameter_ranges: Dict,
                               optimization_params: Dict, output_dir: str) -> str:
    """
    Run a single convergence test with random parameters within specified ranges.

    Parameters:
    -----------
    run_index : int
        Index of the current run
    experiment_data : Dict
        Dictionary containing experiment components (loss_type, instrument_params, spectrum_data)
    parameter_ranges : Dict
        Ranges for random parameter initialization
    optimization_params : Dict
        Gradient descent optimization parameters
    output_dir : str
        Directory to save individual run pickle files

    Returns:
    --------
    str : Path to saved pickle file
    """

    # Reconstruct experiment components from serializable data
    loss_type = experiment_data['loss_type']
    instrument_params = experiment_data['instrument_params']
    spectrum_data = experiment_data['spectrum_data']

    # Recreate instrument
    instrument = GenericSpectrometer(
        instrument_params['resolution'],
        instrument_params['noise']
    )

    # Recreate experimental spectrum
    experimental_spectrum = Spectrum(
        spectrum_data['wavelengths'],
        spectrum_data['intensities']
    )

    # Generate random initial parameters within specified ranges
    temp_range = parameter_ranges.get('temperature', (8000, 12000))
    density_range = parameter_ranges.get('electron_density', (1e22, 5e23))
    length_range = parameter_ranges.get('length', (1e-5, 5e-4))

    initial_temperature = random.uniform(*temp_range)
    initial_density = random.uniform(*density_range)
    initial_length = random.uniform(*length_range)

    # Generate random element ratios within specified ranges
    element_ranges = parameter_ranges.get('element_ratios', {})
    initial_element_ratio = {}

    for element, element_range in element_ranges.items():
        initial_element_ratio[element] = random.uniform(*element_range)

    # Normalize element ratios to sum to 1
    total = sum(initial_element_ratio.values())
    if total > 0:
        for element in initial_element_ratio:
            initial_element_ratio[element] /= total

    # Create initial model with random parameters
    initial_model = LTEModel(
        initial_element_ratio,
        initial_temperature,
        initial_density,
        initial_length
    )

        # Create new experiment with random initial model but same instrument and spectrum
    test_experiment = Experiment(
        loss_type,
        initial_model,
        instrument,
        experimental_spectrum
    )

    print(f"Run {run_index + 1}: T={initial_temperature:.0f}K, "
          f"Ne={initial_density:.2e}m⁻³, L={initial_length:.2e}m")

    # Run gradient descent optimization
    results = custom_gradient_descent(
        experiment=test_experiment,
        learning_rate=optimization_params.get('learning_rate', 5e-5),
        max_iterations=optimization_params.get('max_iterations', 500),
        parameter_change_threshold=optimization_params.get('parameter_change_threshold', 0.001),
        parameter_change_patience=optimization_params.get('parameter_change_patience', 4),
        verbose=False,
        lr_decay_factor=optimization_params.get('lr_decay_factor', 0.8),
        lr_decay_patience=optimization_params.get('lr_decay_patience', 10)
    )

    # Extract model parameters for pickling (LTEModel objects can't be pickled)
    final_model = results['model']

    # Create pickle-safe results
    pickle_safe_results = {
        'run_index': run_index,
        'converged': results['converged'],
        'iterations': results['iterations'],
        'final_loss': results['final_loss'],
        'history': results['history'],
        'initial_parameters': {
            'temperature': initial_temperature,
            'electron_density': initial_density,
            'length': initial_length,
            'element_ratio': initial_element_ratio.copy()
        },
        'final_parameters': {
            'temperature': final_model.temperature,
            'electron_density': final_model.electron_density,
            'length': final_model.length,
            'element_ratio': dict(final_model.element_ratio)
        }
    }

    print(f"Run {run_index + 1}: {'✓ Converged' if results['converged'] else '✗ Did not converge'} "
          f"in {results['iterations']} iterations")

    # Save results to pickle file
    Path(output_dir).mkdir(exist_ok=True)
    output_filename = Path(output_dir) / f"run_{run_index+1:03d}.pkl"

    with open(output_filename, 'wb') as f:
        pickle.dump(pickle_safe_results, f)

    print(f"Run {run_index + 1}: Results saved to {output_filename}")

    return str(output_filename)


def run_convergence_test(experiment: Experiment, parameter_ranges: Dict,
                        num_starting_points: int = 32,
                        output_dir: str = 'convergence_runs',
                        num_processes: Optional[int] = None,
                        optimization_params: Optional[Dict] = None,
                        instrument_params: Optional[Dict] = None) -> Dict:
    """
    Run convergence test with multiple random starting points.

    Parameters:
    -----------
    experiment : Experiment
        The base experiment object (instrument and spectrum will be reused)
    parameter_ranges : Dict
        Ranges for random parameter initialization:
        {
            'temperature': (min, max),
            'electron_density': (min, max),
            'length': (min, max),
            'element_ratios': {
                'Na': (min, max),
                'Mn': (min, max),
                ...

    Examples
    --------
    >>> results = run_convergence_test(
    ...     experiment,
    ...     parameter_ranges={
    ...         'temperature': (5000, 20000),
    ...         'electron_density': (1e22, 1e24),
    ...         'length': (1e-5, 1e-2),
    ...         'element_ratios': {'Fe': (0.3, 1.0), 'Ca': (0.0, 0.7)}
    ...     },
    ...     num_starting_points=16,
    ...     output_dir='convergence_runs'
    ... )
            }
        }
    num_starting_points : int
        Number of different random starting points to test
    output_dir : str
        Directory to save individual run pickle files
    num_processes : int or None
        Number of parallel processes (defaults to CPU count)
    optimization_params : Dict or None
        Gradient descent optimization parameters
    instrument_params : Dict or None
        Instrument parameters if needed (defaults to standard values)

    Returns:
    --------
    Dict : Summary of convergence test results
    """

    if instrument_params is None:
        instrument_params = {'resolution': 0.05, 'noise': 0.001}

    if optimization_params is None:
        optimization_params = {
            'learning_rate': 5e-5,
            'max_iterations': 500,
            'parameter_change_threshold': 0.001,
            'parameter_change_patience': 4,
            'lr_decay_factor': 0.8,
            'lr_decay_patience': 10
        }

    if num_processes is None:
        num_processes = min(mp.cpu_count(), num_starting_points)

    print(f"Running {num_starting_points} convergence tests in parallel using {num_processes} processes...")
    print("=" * 80)

    # Extract serializable data from experiment for multiprocessing
    experiment_data = {
        'loss_type': experiment.loss_type,  # Enum should be pickleable
        'instrument_params': {
            'resolution': experiment.instrument.sigma,  # Use sigma as resolution
            'noise': instrument_params.get('noise', 0.001)  # Use provided noise or default
        },
        'spectrum_data': {
            'wavelengths': np.array(experiment.true_spectrum.wavelengths),
            'intensities': np.array(experiment.true_spectrum.irradiance)
        }
    }

    # Create partial function with fixed parameters
    run_test_partial = partial(
        run_single_convergence_test,
        experiment_data=experiment_data,
        parameter_ranges=parameter_ranges,
        optimization_params=optimization_params,
        output_dir=output_dir
    )

    # Run processing (parallel or sequential based on number of processes)
    if num_processes == 1 or num_starting_points <= 4:
        # Run sequentially to avoid pickling issues
        pickle_files = []
        for i in range(num_starting_points):
            result_file = run_single_convergence_test(
                i, experiment_data,
                parameter_ranges, optimization_params, output_dir
            )
            pickle_files.append(result_file)
    else:
        # Run in parallel (may have pickling issues with some objects)
        with mp.Pool(processes=num_processes) as pool:
            pickle_files = pool.map(run_test_partial, range(num_starting_points))

    print(f"\n" + "=" * 80)
    print(f"CONVERGENCE TEST COMPLETED")
    print(f"=" * 80)
    print(f"Results saved to {len(pickle_files)} individual pickle files in '{output_dir}/'")

    # Return summary
    summary = {
        'num_runs': num_starting_points,
        'output_directory': output_dir,
        'pickle_files': pickle_files,
        'parameter_ranges': parameter_ranges,
        'optimization_params': optimization_params
    }

    return summary


def load_convergence_results(output_dir: str) -> list:
    """
    Load all convergence test results from pickle files in a directory.

    Parameters:
    -----------
    output_dir : str
        Directory containing individual run pickle files

    Returns:
    --------
    Dict : All loaded results
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist")

    # Find all pickle files
    pickle_files = sorted(output_path.glob('run_*.pkl'))

    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in '{output_dir}'")

    print(f"Loading {len(pickle_files)} result files from '{output_dir}'...")

    results = []

    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                result = pickle.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {pickle_file}: {e}")

    return results


def plot_convergence_results(convergence_results, true_model, save_plots=False, output_dir='convergence_plots'):
    """
    Generate comprehensive plots from convergence test results.

    Parameters:
    -----------
    results : list
        Results from load_convergence_results (list of individual run results)
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plots in
    true_model : LTEModel or None
        True model parameters for comparison (optional)

    Returns:
    --------
    None
    """

    # Extract convergence data
    histories = [res['history'] for res in convergence_results]
    converged = [res['converged'] for res in convergence_results]
    iterations = [res['iterations'] for res in convergence_results]
    num_starting_points = len(convergence_results)

    true_model_params = {
        'temperature': true_model.temperature,
        'electron_density': true_model.electron_density,
        'length': true_model.length,
        'element_ratio': dict(true_model.element_ratio)
    }

    true_element_ratio = true_model_params['element_ratio']

    # Create output directory if saving plots
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)

    # Create comprehensive visualization for convergence test results
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # Color palette for different runs
    colors = cm.get_cmap('gist_rainbow')(np.linspace(0, 1, num_starting_points))

    # 1. Loss convergence for all runs
    for i, history in enumerate(histories):
        loss_iterations = np.arange(len(history['loss']))
        axes[0, 0].semilogy(loss_iterations, history['loss'],
                           color=colors[i], linewidth=1.5, alpha=0.7,
                           label=f'Run {i+1}' if i < 5 else "")
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].set_title('Loss Convergence - All Starting Points')
    axes[0, 0].grid(True, alpha=0.3)
    if num_starting_points <= 5:
        axes[0, 0].legend()

    # 4. Temperature convergence paths
    for i, history in enumerate(histories):
        param_iterations = np.arange(len(history['temperature']))
        axes[1, 0].plot(param_iterations, np.array(history['temperature']),
                       color=colors[i], linewidth=1.5, alpha=0.7,
                       label=f'Run {i+1}' if i < 5 else "")
    axes[1, 0].axhline(y=true_model_params['temperature'], color='red', linestyle='--',
                      linewidth=2, alpha=0.9, label='True Value')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Temperature (K)')
    axes[1, 0].set_title('Temperature Convergence - All Runs')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Electron density convergence paths
    for i, history in enumerate(histories):
        param_iterations = np.arange(len(history['electron_density']))
        axes[1, 1].semilogy(param_iterations, np.array(history['electron_density']),
                       color=colors[i], linewidth=1.5, alpha=0.7,
                       label=f'Run {i+1}' if i < 5 else "")
    axes[1, 1].axhline(y=true_model_params['electron_density'], color='red', linestyle='--',
                      linewidth=2, alpha=0.9, label='True Value')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Electron Density (m$^{-3}$)')
    axes[1, 1].set_title('Electron Density Convergence - All Runs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Length convergence paths
    for i, history in enumerate(histories):
        param_iterations = np.arange(len(history['length']))
        axes[1, 2].semilogy(param_iterations, np.array(history['length']),
                       color=colors[i], linewidth=1.5, alpha=0.7,
                       label=f'Run {i+1}' if i < 5 else "")
    axes[1, 2].axhline(y=true_model_params['length'], color='red', linestyle='--',
                      linewidth=2, alpha=0.9, label='True Value')
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Length (m)')
    axes[1, 2].set_title('Length Convergence - All Runs')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # 7. Element ratio convergence - Na and Mn
    for i, history in enumerate(histories):
        param_iterations = np.arange(len(history['Na']))
        axes[2, 0].plot(param_iterations, history['Na'],
                       color=colors[i], linewidth=1.5, alpha=0.7, linestyle='-')
        axes[2, 0].plot(param_iterations, history['Mn'],
                       color=colors[i], linewidth=1.5, alpha=0.7, linestyle='--')

    axes[2, 0].axhline(y=true_element_ratio['Na'], color='blue', linestyle='-',
                      linewidth=2, alpha=0.9, label='True Na')
    axes[2, 0].axhline(y=true_element_ratio['Mn'], color='red', linestyle='--',
                      linewidth=2, alpha=0.9, label='True Mn')
    axes[2, 0].set_xlabel('Iteration')
    axes[2, 0].set_ylabel('Element Ratio')
    axes[2, 0].set_title('Na and Mn Convergence - All Runs')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 8. Element ratio convergence - Mg and Ca
    for i, history in enumerate(histories):
        param_iterations = np.arange(len(history['Mg']))
        axes[2, 1].plot(param_iterations, history['Mg'],
                       color=colors[i], linewidth=1.5, alpha=0.7, linestyle='-')
        axes[2, 1].plot(param_iterations, history['Ca'],
                       color=colors[i], linewidth=1.5, alpha=0.7, linestyle='--')

    axes[2, 1].axhline(y=true_element_ratio['Mg'], color='green', linestyle='-',
                      linewidth=2, alpha=0.9, label='True Mg')
    axes[2, 1].axhline(y=true_element_ratio['Ca'], color='orange', linestyle='--',
                      linewidth=2, alpha=0.9, label='True Ca')
    axes[2, 1].set_xlabel('Iteration')
    axes[2, 1].set_ylabel('Element Ratio')
    axes[2, 1].set_title('Mg and Ca Convergence - All Runs')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # 9. Final parameter recovery accuracy summary
    elements = list(true_element_ratio.keys())
    final_element_errors = {}
    for element in elements:
        final_element_errors[element] = [abs(history[element][-1] - true_element_ratio[element]) / true_element_ratio[element] * 100
                                        for history in histories]

    # Box plot of final errors
    final_temp_errors = [abs(history['temperature'][-1] - true_model_params['temperature']) / true_model_params['temperature'] * 100
                        for history in histories]
    final_density_errors = [abs(history['electron_density'][-1] - true_model_params['electron_density']) / true_model_params['electron_density'] * 100
                           for history in histories]
    final_length_errors = [abs(history['length'][-1] - true_model_params['length']) / true_model_params['length'] * 100
                          for history in histories]
    all_errors = [final_temp_errors, final_density_errors, final_length_errors] + [final_element_errors[el] for el in elements]
    error_labels = ['Temp', 'Density', 'Length'] + elements

    box_plot = axes[2, 2].boxplot(all_errors, labels=error_labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], ['orange', 'green', 'purple', 'blue', 'red', 'darkgreen', 'darkorange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[2, 2].set_ylabel('Relative Error (%)')
    axes[2, 2].set_title('Parameter Recovery Error Distribution')
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f'{output_dir}/convergence_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/convergence_overview.pdf', bbox_inches='tight')

    plt.show()

    # Print summary statistics
    print("\n" + "="*80)
    print("CONVERGENCE TEST SUMMARY STATISTICS")
    print("="*80)

    print(f"Total runs: {num_starting_points}")
    print(f"Successful convergences: {sum(converged)} ({sum(converged)/num_starting_points*100:.1f}%)")
    print(f"Average iterations to convergence: {np.mean([it for it, conv in zip(iterations, converged) if conv]):.1f}")

    print(f"\nParameter recovery statistics:")
    print(f"Temperature errors: {np.mean(final_temp_errors):.1f}% ± {np.std(final_temp_errors):.1f}%")
    print(f"Density errors: {np.mean(final_density_errors):.1f}% ± {np.std(final_density_errors):.1f}%")
    print(f"Length errors: {np.mean(final_length_errors):.1f}% ± {np.std(final_length_errors):.1f}%")

    for element in elements:
        element_errors = final_element_errors[element]
        print(f"{element} errors: {np.mean(element_errors):.1f}% ± {np.std(element_errors):.1f}%")

    print(f"\nBest performing run: {np.argmin([np.mean([final_temp_errors[i], final_density_errors[i], final_length_errors[i]]) for i in range(num_starting_points)]) + 1}")
    print(f"Worst performing run: {np.argmax([np.mean([final_temp_errors[i], final_density_errors[i], final_length_errors[i]]) for i in range(num_starting_points)]) + 1}")

    # Robustness assessment
    overall_errors = [np.mean([final_temp_errors[i], final_density_errors[i], final_length_errors[i]]) for i in range(num_starting_points)]
    robustness_score = (1 - np.std(overall_errors)/np.mean(overall_errors)) * 100
    print(f"\nAlgorithm robustness score: {robustness_score:.1f}% (higher = more consistent across starting points)")

    success_rate = sum(converged)/num_starting_points*100
    if success_rate >= 80:
        print(f"✓ Excellent convergence reliability ({success_rate:.1f}%)")
    elif success_rate >= 60:
        print(f"⚠ Good convergence reliability ({success_rate:.1f}%)")
    else:
        print(f"⚠ Poor convergence reliability ({success_rate:.1f}%)")
