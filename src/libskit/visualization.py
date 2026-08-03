#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Visualization functions for libskit optimization results.

This module provides comprehensive plotting capabilities for analyzing and visualizing
the results of optimization runs (both the new :class:`Optimizer` and the legacy
``custom_gradient_descent``).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib import animation
from typing import Dict, Any, List, Optional, Tuple
from ._libskit import LTEModel


# ---------------------------------------------------------------------------
# New generic plotting for Optimizer results
# ---------------------------------------------------------------------------

def plot_optimization_results(
    result: Dict[str, Any],
    experiment,
    true_model=None,
    figsize: tuple = (18, 12),
) -> tuple:
    """Visualise results from :class:`Optimizer.optimize`.

    Creates a multi-panel figure showing loss convergence, spectrum
    comparison, residuals, and per-parameter evolution.  Works with
    **any** plasma model type.

    Parameters
    ----------
    result : dict
        Dictionary returned by :meth:`Optimizer.optimize`.
    experiment : Experiment
        The libskit experiment.
    true_model : PlasmaModel, optional
        "Ground-truth" model for comparison lines.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    tuple
        ``(fig, axes)`` – matplotlib figure and axes array.
    """
    history = result["history"]
    model = experiment.get_plasma_model()
    param_names = list(model.get_parameter_names())
    instrument = experiment.get_instrument()
    wavelengths = experiment.true_spectrum.wavelengths
    true_spectrum = experiment.true_spectrum.irradiance

    # Separate physical params and element params
    phys_params = [n for n in param_names if "_" in n or n in ("temperature", "electron_density", "length")]
    elem_params = [n for n in param_names if n not in phys_params]

    # We need: loss + spectrum + residuals + phys params + element params
    n_phys = len(phys_params)
    n_panels = 3 + n_phys + (1 if elem_params else 0)
    ncols = 3
    nrows = max(2, (n_panels + ncols - 1) // ncols)

    fig, axes_2d = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes_2d.flat  # flat iterator

    # ---- Panel 0: Loss convergence ----
    ax = axes[0]
    ax.semilogy(history["loss"], "b-", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Convergence")
    ax.grid(True, alpha=0.3)

    # ---- Panel 1: Spectrum comparison ----
    ax = axes[1]
    ax.plot(wavelengths, true_spectrum, "b-", linewidth=1, alpha=0.7, label="Data")
    final_spectrum = instrument.calculate_spectrum(wavelengths, model)
    ax.plot(wavelengths, final_spectrum, "r-", linewidth=1, alpha=0.7, label="Fit")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Spectrum Comparison")

    # ---- Panel 2: Residuals ----
    ax = axes[2]
    residuals = np.asarray(final_spectrum) - np.asarray(true_spectrum)
    ax.plot(wavelengths, residuals, "g-", linewidth=1, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)
    ax.set_title("Fit Residuals")

    # ---- Panels for physical parameters ----
    panel_idx = 3
    for name in phys_params:
        if name not in history:
            continue
        ax = axes[panel_idx]
        ax.plot(history[name], linewidth=2)
        if true_model is not None:
            try:
                true_val = true_model.get_parameter(name)
                ax.axhline(y=true_val, color="red", linestyle="--", alpha=0.7, label="True")
                ax.legend()
            except Exception:
                pass
        ax.set_xlabel("Iteration")
        ax.set_ylabel(name)
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # ---- Panel for element ratios ----
    if elem_params:
        ax = axes[panel_idx]
        colors = plt.cm.tab10.colors
        for j, name in enumerate(elem_params):
            if name not in history:
                continue
            c = colors[j % len(colors)]
            ax.plot(history[name], color=c, linewidth=2, label=name)
            if true_model is not None:
                try:
                    true_val = true_model.get_parameter(name)
                    ax.axhline(y=true_val, color=c, linestyle="--", alpha=0.7)
                except Exception:
                    pass
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative Abundance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Element Ratios")
        panel_idx += 1

    # Turn off unused axes
    for k in range(panel_idx, nrows * ncols):
        axes[k].axis("off")

    fig.suptitle(
        f"{result.get('algorithm', 'Optimizer')} — "
        f"final loss {result['final_loss']:.3e} — "
        f"{result['n_evaluations']} evals",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig, axes_2d


# ---------------------------------------------------------------------------
# Legacy plotting functions (for custom_gradient_descent compatibility)
# ---------------------------------------------------------------------------


def plot_gradient_descent_results(gd_results: Dict[str, Any], experiment, true_model: LTEModel,
                                figsize: tuple = (18, 15)) -> tuple:
    """
    Create comprehensive visualization of gradient descent optimization results.

    Parameters:
    -----------
    gd_results : dict
        Results dictionary from custom_gradient_descent function containing:
        - 'model': final optimized model
        - 'history': optimization history with parameter evolution
    experiment : libskit.Experiment
        The experiment object containing spectrum and instrument data
    true_model : LTEModel
        The true/target model used for comparison
    figsize : tuple, optional
        Figure size (width, height) in inches, default (18, 15)

    Returns:
    --------
    tuple: (fig, axes)
        Matplotlib figure and axes objects for further customization
    """

    # Extract data from experiment
    wavelengths = experiment.true_spectrum.wavelengths
    true_spectrum = experiment.true_spectrum.irradiance
    instrument = experiment.get_instrument()
    true_element_ratio = dict(true_model.element_ratio)

    # Extract data from results
    final_model = gd_results['model']
    element_histories = {el: gd_results['history'][el] for el in final_model.element_ratio.keys()}
    temperatures = gd_results['history']['temperature']
    electron_densities = gd_results['history']['electron_density']
    lengths = gd_results['history']['length']
    loss_history = gd_results['history']['loss']

    # Create visualization of optimization results
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    ##########################
    # Plot 1 - Loss function #
    ##########################
    loss_iterations = np.arange(len(gd_results['history']['loss']))
    axes[0, 0].semilogy(loss_iterations, loss_history, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].set_title('Loss Function Convergence')
    axes[0, 0].grid(True, alpha=0.3)

    ################################
    # Plot 2 - Spectrum Comparison #
    ################################
    axes[0, 1].plot(wavelengths, true_spectrum, 'b-', linewidth=1, alpha=0.7, label='True Spectrum')
    final_spectrum = instrument.calculate_spectrum(wavelengths, final_model)
    axes[0, 1].plot(wavelengths, final_spectrum, 'r-', linewidth=1, alpha=0.7, label='Fitted Spectrum')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(250, 900)
    axes[0, 1].set_title('Spectrum Comparison')

    ######################
    # Plot 3 - Residuals #
    ######################
    residuals = final_spectrum- true_spectrum
    axes[0, 2].plot(wavelengths, residuals, 'g-', linewidth=1, alpha=0.7)
    axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    axes[0, 2].set_xlabel('Wavelength (nm)')
    axes[0, 2].set_ylabel('Residuals (Fitted - True)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(250, 900)
    axes[0, 2].set_title('Fit Residuals')

    #########################################
    # Plot 4 - Element ratios per iteration #
    #########################################
    param_iterations = np.arange(len(temperatures))
    colors = ['blue', 'red', 'green', 'orange']
    element_names = ["Na", "Mn", "Mg", "Ca"]

    for i, element in enumerate(element_names):
        if element in element_histories:
            axes[1, 0].plot(param_iterations[:len(element_histories[element])],
                           element_histories[element],
                           color=colors[i], linewidth=2, label=f'{element}')
            if element in true_element_ratio:
                axes[1, 0].axhline(y=true_element_ratio[element],
                                  color=colors[i], linestyle='--', alpha=0.7)

    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Relative Abundance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Element Ratio Evolution')

    ######################################
    # Plot 5 - Temperature per iteration #
    ######################################
    axes[1, 1].plot(param_iterations, np.array(temperatures), label='Temperature (K)', linewidth=2)
    axes[1, 1].axhline(y=true_model.temperature, color='red', linestyle='--', alpha=0.7, label='True Temperature')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Temperature (K)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Temperature Evolution')

    ####################################
    # Plot 6 - n_e and L per iteration #
    ####################################
    ax_density = axes[1, 2]
    ax_length = ax_density.twinx()

    line1 = ax_density.plot(param_iterations, np.array(electron_densities), 'g-', linewidth=2, label='Electron Density ($ m^{-3}$)')
    ax_density.axhline(y=true_model.electron_density, color='green', linestyle='--', alpha=0.7)

    line2 = ax_length.plot(param_iterations, np.array(lengths), 'purple', linewidth=2, label='Length (m)')
    ax_length.axhline(y=true_model.length, color='purple', linestyle='--', alpha=0.7)

    ax_density.set_xlabel('Iteration')
    ax_density.set_ylabel('Electron Density ($ m^{-3}$)', color='green')
    ax_length.set_ylabel('Length ($ m$)', color='purple')
    ax_density.grid(True, alpha=0.3)
    ax_density.set_title('Density & Length Evolution')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_density.legend(lines, labels, loc='upper right')

    ####################################
    # Plot 7 - Quality Metrics        #
    ####################################
    # RMSE and R-squared on left axis
    ax_quality = axes[2, 0]
    ax_rsquared = ax_quality.twinx()

    quality_iterations = np.arange(len(gd_results['history']['rmse']))

    line1 = ax_quality.semilogy(quality_iterations, gd_results['history']['rmse'], 'r-', linewidth=2, label='RMSE')
    line2 = ax_quality.semilogy(quality_iterations, gd_results['history']['chi_squared'], 'b-', linewidth=2, label='Chi2')
    line3 = ax_rsquared.plot(quality_iterations, gd_results['history']['r_squared'], 'g-', linewidth=2, label='R2')

    ax_quality.set_xlabel('Iteration')
    ax_quality.set_ylabel('RMSE / Chi2', color='black')
    ax_rsquared.set_ylabel('R2 (Coefficient of Determination)', color='green')
    ax_quality.grid(True, alpha=0.3)
    ax_quality.set_title('Quality Metrics Evolution')

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax_quality.legend(lines, labels, loc='upper right')

    # Plot 8 & 9 - Additional space for future metrics or custom plots
    axes[2, 1].axis('off')  # Turn off empty subplot
    axes[2, 2].axis('off')  # Turn off empty subplot

    plt.tight_layout()
    return fig, axes


def create_gradient_descent_animation(gd_results: Dict[str, Any], experiment, true_model: LTEModel,
                                    wavelengths: np.ndarray, sodium_range: Tuple[float, float] = (580, 620),
                                    skip_frames: int = 3) -> Tuple[animation.FuncAnimation, figure.Figure]:
    """
    Create a comprehensive animation showing gradient descent optimization progress.

    This function creates an interactive animation that displays:
    1. Full spectrum evolution during optimization
    2. Focused view of sodium spectral lines
    3. Residuals between true and current guess spectra
    4. Sodium parameter evolution over iterations
    5. Loss function evolution over iterations

    Parameters:
    -----------
    gd_results : Dict[str, Any]
        Results dictionary from custom_gradient_descent function containing:
        - 'model': final optimized model
        - 'history': optimization history with parameter evolution
    experiment : libskit.Experiment
        The experiment object containing spectrum and instrument data
    true_model : LTEModel
        The true/target model used for comparison
    wavelengths : np.ndarray
        Array of wavelengths for spectrum calculation
    sodium_range : Tuple[float, float], optional
        Wavelength range for sodium line focus, default (580, 620) nm
    skip_frames : int, optional
        Number of frames to skip for smoother animation, default 3

    Returns:
    --------
    Tuple[animation.FuncAnimation, plt.Figure]
        Matplotlib animation object and figure for display and further customization

    Example:
    --------
    >>> anim, fig = create_gradient_descent_animation(results, experiment, true_model, wavelengths)
    >>> plt.show()  # Display the animation
    >>> anim.save('optimization.gif', writer='pillow', fps=3)  # Save as GIF
    """

    history = gd_results['history']
    n_iterations = len(history['loss'])

    # Extract true element ratios for comparison
    true_element_ratio = dict(true_model.element_ratio)

    # Get the true spectrum for comparison
    true_spectrum = experiment.get_instrument().calculate_spectrum(wavelengths, true_model)

    # Create adaptive sampling with three phases for better animation flow
    iteration_indices = []

    # Phase 1: First 20 iterations - show every iteration (detailed early progress)
    early_iterations = min(20, n_iterations)
    iteration_indices.extend(range(0, early_iterations))

    # Phase 2: Iterations 20-100 - use normal skip_frames (medium detail)
    if n_iterations > 20:
        mid_end = min(100, n_iterations)
        mid_iterations = list(range(20, mid_end, skip_frames))
        iteration_indices.extend(mid_iterations)

    # Phase 3: Iterations 100+ - use larger skip for faster animation (convergence phase)
    if n_iterations > 100:
        fast_skip = skip_frames * 2
        late_iterations = list(range(100, n_iterations, fast_skip))
        iteration_indices.extend(late_iterations)

    # Always include the final iteration
    if (n_iterations - 1) not in iteration_indices:
        iteration_indices.append(n_iterations - 1)

    # Remove duplicates and sort
    iteration_indices = sorted(list(set(iteration_indices)))

    # Create masks for sodium region focus
    sodium_mask = (wavelengths >= sodium_range[0]) & (wavelengths <= sodium_range[1])
    wl_sodium = wavelengths[sodium_mask]
    true_spectrum_sodium = true_spectrum.real[sodium_mask]

    # Pre-calculate all spectra for animation (improves performance)
    spectra_evolution = []
    spectra_sodium = []
    residuals_full = []
    residuals_sodium = []

    print(f"Calculating animation data for {len(iteration_indices)} frames...")

    for i, iter_idx in enumerate(iteration_indices):
        # Reconstruct model at this iteration
        element_ratio = {element: history[element][iter_idx] for element in true_element_ratio.keys()}
        iter_model = LTEModel(
            element_ratio,
            history['temperature'][iter_idx],
            history['electron_density'][iter_idx],
            history['length'][iter_idx]
        )

        # Calculate full spectrum at this iteration
        iter_spectrum = experiment.get_instrument().calculate_spectrum(wavelengths, iter_model)
        iter_spectrum_real = iter_spectrum.real

        # Store full spectrum and sodium region data
        spectra_evolution.append(iter_spectrum_real)
        spectra_sodium.append(iter_spectrum_real[sodium_mask])

        # Calculate residuals (difference between true and current guess)
        residual_full = true_spectrum.real - iter_spectrum_real
        residual_sodium = true_spectrum_sodium - iter_spectrum_real[sodium_mask]
        residuals_full.append(residual_full)
        residuals_sodium.append(residual_sodium)

        # Progress indicator
        if i % 10 == 0:
            print(f"  Processed frame {i+1}/{len(iteration_indices)}")

    # Create the comprehensive figure with 4 subplots (2x2 grid)
    fig = plt.figure(figsize=(20, 10))

    # Create subplot layout with proper spacing
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                         top=0.90, hspace=0.3, wspace=0.3)

    # Subplot 1: Full spectrum view (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    line_true_full, = ax1.plot(wavelengths, true_spectrum.real, 'b-', linewidth=1.5,
                              label='True Spectrum', alpha=0.8)
    line_guess_full, = ax1.plot(wavelengths, spectra_evolution[0], 'r-', linewidth=1.5,
                               label='Current Guess')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Irradiance (W/m²/nm/sr)')
    ax1.set_title('Full Spectrum Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Set consistent y-limits for full spectrum
    all_spectra = np.array(spectra_evolution + [true_spectrum.real])
    y_min_full, y_max_full = np.min(all_spectra), np.max(all_spectra)
    ax1.set_ylim(y_min_full * 0.95, y_max_full * 1.05)

    # Subplot 2: Sodium region focus (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    line_true_sodium, = ax2.plot(wl_sodium, true_spectrum_sodium, 'b-', linewidth=2,
                                label='True Spectrum', alpha=0.8)
    line_guess_sodium, = ax2.plot(wl_sodium, spectra_sodium[0], 'r-', linewidth=2,
                                 label='Current Guess')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Irradiance (W/m²/nm/sr)')
    ax2.set_title(f'Sodium Lines Focus: {sodium_range[0]}-{sodium_range[1]} nm')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Set consistent y-limits for sodium region
    all_sodium = np.array(spectra_sodium + [true_spectrum_sodium])
    y_min_sodium, y_max_sodium = np.min(all_sodium), np.max(all_sodium)
    ax2.set_ylim(y_min_sodium * 0.95, y_max_sodium * 1.05)

    # Subplot 3: Full spectrum residuals (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    line_residual_full, = ax3.plot(wavelengths, residuals_full[0], 'g-', linewidth=1,
                                  label='Residuals (True - Guess)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Residual Irradiance')
    ax3.set_title('Full Spectrum Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Combined sodium parameter evolution and loss (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])

    # Sodium parameter evolution (primary y-axis)
    color_na = 'purple'
    ax4.plot(range(n_iterations), history['Na'], color_na, alpha=0.7, linewidth=1.5,
             label='Na Evolution')
    ax4.axhline(y=true_element_ratio['Na'], color='b', linestyle='--', alpha=0.8,
               label=f'True Na ({true_element_ratio["Na"]:.2f})')
    line_na_marker, = ax4.plot(iteration_indices[0], history['Na'][iteration_indices[0]],
                              'o', color=color_na, markersize=8, label='Current Na Value')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Na Fraction', color=color_na)
    ax4.tick_params(axis='y', labelcolor=color_na)
    ax4.grid(True, alpha=0.3)

    # Create secondary y-axis for loss function
    ax4_twin = ax4.twinx()
    color_loss = 'red'
    ax4_twin.plot(range(n_iterations), history['loss'], color_loss, alpha=0.7, linewidth=1.5)
    line_loss_marker, = ax4_twin.plot(iteration_indices[0], history['loss'][iteration_indices[0]],
                                     'o', color=color_loss, markersize=8, label='Current Loss')
    ax4_twin.set_ylabel('Loss', color=color_loss)
    ax4_twin.set_yscale('log')
    ax4_twin.tick_params(axis='y', labelcolor=color_loss)

    ax4.set_title('Sodium Parameter & Loss Evolution')

    # Combine legends from both axes
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # Animation update function
    def animate_comprehensive(frame):
        """Update function called for each animation frame."""
        iter_idx = iteration_indices[frame]

        # Update spectrum lines and residuals
        line_guess_full.set_ydata(spectra_evolution[frame])
        line_guess_sodium.set_ydata(spectra_sodium[frame])
        line_residual_full.set_ydata(residuals_full[frame])

        # Update parameter and loss markers
        line_na_marker.set_data([iter_idx], [history['Na'][iter_idx]])
        line_loss_marker.set_data([iter_idx], [history['loss'][iter_idx]])

        # Update main title with current iteration info
        loss_val = history['loss'][iter_idx]
        na_val = history['Na'][iter_idx]

        fig.suptitle(f'Iteration {iter_idx} | Loss: {loss_val:.2e} | Na: {na_val:.3f}',
                    fontsize=14, fontweight='bold', y=0.95)

        return (line_guess_full, line_guess_sodium, line_residual_full,
                line_na_marker, line_loss_marker)

    # Create animation with adaptive timing
    anim = animation.FuncAnimation(
        fig, animate_comprehensive, frames=len(iteration_indices),
        interval=300, blit=True, repeat=True
    )

    # Note: Animation uses consistent interval for all frames
    # Advanced adaptive timing could be implemented by modifying interval during runtime

    plt.tight_layout()
    return anim, fig
