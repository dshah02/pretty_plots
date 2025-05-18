import numpy as np
import matplotlib.pyplot as plt
from pretty_plots import PrettyPlots as pp
import os

# Create output directory for saved figures
os.makedirs('figures', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Line Plot Example
# ============================================================================
print("1. Demonstrating Line Plot...")

# Generate test data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# Create and display line plot
fig, ax = pp.line_plot(
    x_values=[x, x, x],
    y_values=[y1, y2, y3],
    labels=["sin(x)", "cos(x)", "sin(x)cos(x)"],
    title="Trigonometric Functions",
    xlabel="x",
    ylabel="f(x)",
    style_name="research_paper"
)

# Save the figure
pp.save_figure(fig, "figures/1_line_plot", formats=["png"])
# plt.show()

# ============================================================================
# 2. Multi-Line Comparison Example
# ============================================================================
print("\n2. Demonstrating Multi-Line Comparison...")

# Generate test data for two groups of lines
x = np.linspace(0, 10, 100)

# Data for two subplots
y1_1 = np.sin(x)
y1_2 = np.sin(x + np.pi/4)
y1_3 = np.sin(x + np.pi/2)

y2_1 = np.exp(-0.1*x) * np.cos(x)
y2_2 = np.exp(-0.2*x) * np.cos(x)
y2_3 = np.exp(-0.3*x) * np.cos(x)

# Create and display multi-line comparison plot
fig, axes = pp.multi_line_comparison(
    x_values=[[x, x, x], [x, x, x]],
    y_values=[[y1_1, y1_2, y1_3], [y2_1, y2_2, y2_3]],
    subplot_titles=["Sine Functions with Phase Shift", "Damped Cosine Functions"],
    labels=["$\\sin(x)$", "$\\sin(x + \\pi/4)$", "$\\sin(x + \\pi/2)$"],
    main_title="Wave Function Comparison",
    xlabel="x",
    ylabel="f(x)",
    style_name="research_paper"
)

# Save the figure
pp.save_figure(fig, "figures/2_multi_line_comparison", formats=["png"])
# plt.show()

# ============================================================================
# 3. Bar Plot Example
# ============================================================================
print("\n3. Demonstrating Bar Plot...")

# Generate test data
categories = ["Category A", "Category B", "Category C", "Category D"]
values = [
    [4.2, 3.8, 5.1, 4.5],  # Group 1
    [3.5, 4.5, 4.2, 3.9],  # Group 2
    [5.0, 3.2, 4.8, 5.2]   # Group 3
]

# Create and display bar plot
fig, ax = pp.bar_plot(
    categories=categories,
    values=values,
    group_labels=["Method 1", "Method 2", "Method 3"],
    title="Performance Comparison Across Categories",
    xlabel="Test Categories",
    ylabel="Performance Score",
    show_values=True,
    style_name="bar_chart"
)

# Save the figure
pp.save_figure(fig, "figures/3_bar_plot", formats=["png"])

# ============================================================================
# 4. Histogram Example
# ============================================================================
print("\n4. Demonstrating Histogram...")

# Generate test data from different distributions
data = np.random.normal(0, 1, 1000)

# Create and display histogram
fig, ax = pp.histogram(
    data=data,
    title="Distribution of Test Scores",
    xlabel="Score",
    ylabel="Frequency",
    bins=30,
    color="#4C72B0",
    style_name="histogram"
)

# Save the figure
pp.save_figure(fig, "figures/4_histogram", formats=["png"])
pp.show()

# ============================================================================
# 5. Double Histogram Example
# ============================================================================
print("\n5. Demonstrating Double Histogram...")

# Generate test data from different distributions
data1 = np.random.normal(0, 1, 1000)  # Control group
data2 = np.random.normal(0.5, 1.2, 1000)  # Treatment group

# Create and display double histogram
fig, ax = pp.double_histogram(
    data1=data1,
    data2=data2,
    label1="Control Group",
    label2="Treatment Group",
    title="Comparison of Test Results",
    xlabel="Score",
    ylabel="Frequency",
    bins=40,
    style_name="histogram"
)

# Save the figure
pp.save_figure(fig, "figures/5_double_histogram", formats=["png"])
plt.show()

# ============================================================================
# 6. Confidence Plot Example
# ============================================================================
print("\n6. Demonstrating Confidence Plot...")

# Generate test data with confidence intervals
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-0.1 * x)
errors = 0.1 + 0.05 * np.abs(np.sin(3*x))  # Varying uncertainty

# Create and display confidence plot
fig, ax = pp.confidence_plot(
    x_values=x,
    y_values=y,
    error_values=errors,
    title="Damped Sine Wave with Uncertainty",
    xlabel="Time",
    ylabel="Amplitude $\\pm \\sigma$",
    style_name="research_paper"
)

# Save the figure
pp.save_figure(fig, "figures/6_confidence_plot", formats=["png"])

# ============================================================================
# 7. Experiment Plot Example
# ============================================================================
print("\n7. Demonstrating Experiment Plot...")

# Generate test data for multiple experimental runs
n_steps = 100
n_runs = 5

# Create three different "experiments" with multiple runs each
# Experiment 1: Fast convergence but plateaus
exp1_data = [100 * np.exp(-0.05 * np.arange(n_steps)) + 
             10 * np.random.randn(n_steps) for _ in range(n_runs)]

# Experiment 2: Slower convergence but better final result
exp2_data = [80 * np.exp(-0.03 * np.arange(n_steps)) + 
             8 * np.random.randn(n_steps) for _ in range(n_runs)]

# Experiment 3: Slowest convergence
exp3_data = [60 * np.exp(-0.02 * np.arange(n_steps)) + 
             5 * np.random.randn(n_steps) for _ in range(n_runs)]

# Create and display experiment plot
data_dict = {
    "Algorithm A": exp1_data,
    "Algorithm B": exp2_data,
    "Algorithm C": exp3_data
}

fig, ax = pp.experiment_plot(
    data_dict=data_dict,
    title="Training Loss Comparison",
    xlabel="Training Iteration",
    ylabel="Loss",
    smoothing=5,
    ylim=(0, 120),
    style_name="research_paper"
)

# Save the figure
pp.save_figure(fig, "figures/7_experiment_plot", formats=["png"])
plt.show()

# ============================================================================
# 8. Sequence Comparison Example
# ============================================================================
print("\n8. Demonstrating Sequence Comparison...")

# Generate pairs of similar sequences with various patterns
np.random.seed(42)

# First pair - sine waves with noise
seq1_1 = np.sin(np.linspace(0, 6*np.pi, 200))
seq1_2 = seq1_1 + 0.1 * np.random.randn(200)

# Second pair - exponential decay
seq2_1 = np.exp(-0.01 * np.arange(200))
seq2_2 = seq2_1 * (1 + 0.05 * np.random.randn(200))

# Third pair - square wave
t = np.linspace(0, 1, 200)
seq3_1 = np.where(t % 0.25 < 0.125, 1, -1)
seq3_2 = seq3_1 + 0.2 * np.random.randn(200)

# Create and display sequence comparison
fig, axes = pp.sequence_comparison(
    sequences=[seq1_1, seq1_2, seq2_1, seq2_2, seq3_1, seq3_2],
    labels=["Ground Truth", "Prediction", "Ground Truth", "Prediction", "Ground Truth", "Prediction"],
    title="Model Prediction vs Ground Truth",
    xlabel="Time",
    ylabel="Signal Value",
    layout="grid",
    n_cols=2,
    style_name="research_paper"
)

# Save the figure
pp.save_figure(fig, "figures/8_sequence_comparison", formats=["png"])

# ============================================================================
# 9. Triple Subplot Example for Runtime Analysis
# ============================================================================
print("\n9. Demonstrating Triple Subplot for Runtime Analysis...")

# Sample data for runtime comparison
seq_lengths = [4096, 8192, 16384, 32768, 65536, 131072]

# Data for subplots (simulation of different model runtimes)
# Model runtimes for STU-Only
model1_data = [13.74, 27.42, 54.73, 108.68, 215.00, 428.36]  # SpectraLDS
model2_data = [12.14, 24.84, 57.32, 149.12, 429.20, 1352.40]  # Baseline
model3_data = [19.35, 35.86, 67.96, 135.75, 329.91, 952.70]  # Future Fill

# Hybrid model runtimes
hybrid1_data = [20.40, 40.82, 81.63, 163.30, 325.92, 651.35]  # SpectraLDS
hybrid2_data = [18.91, 38.09, 76.49, 164.70, 389.72, 1014.67]  # Baseline
hybrid3_data = [19.92, 38.39, 74.89, 141.16, 290.68, 666.24]  # Future Fill

# Extended sequence lengths for the third plot
extended_lengths = [32768, 65536, 131072, 262144, 524288, 1048576]

# Convolution-only runtimes
conv1_data = [3.69, 7.36, 14.65, 29.16, 58.03, 116.00]  # SpectraLDS (SD 800)
conv2_data = [3.62, 7.18, 14.36, 28.57, 57.08, 114.04]  # SpectraLDS (SD 100)
conv3_data = [4.29, 11.61, 36.43, 117.80, 365.17, 1145.86]  # Baseline
conv4_data = [7.14, 13.49, 27.25, 63.44, 177.50, 576.13]  # Future Fill

# Create and display triple subplot
fig, axes = pp.triple_subplot(
    x_values=[
        [seq_lengths, seq_lengths, seq_lengths], 
        [seq_lengths, seq_lengths, seq_lengths],
        [extended_lengths, extended_lengths, extended_lengths, extended_lengths]
    ],
    y_values=[
        [model1_data, model2_data, model3_data],
        [hybrid1_data, hybrid2_data, hybrid3_data],
        [conv1_data, conv2_data, conv3_data, conv4_data]
    ],
    subplot_titles=["Runtime for STU-Only Models", "Runtime for Hybrid Models", "Runtime for Convolution Only"],
    labels=[
        ["SpectraLDS", "Baseline", "Future Fill"],
        ["SpectraLDS", "Baseline", "Future Fill"],
        ["SpectraLDS (SD 800)", "SpectraLDS (SD 100)", "Baseline", "Future Fill"]
    ],
    xlabel="Sequence Length (tokens)",
    ylabel="Runtime (seconds)",
    xscale="log2",
    xlim=[(4000, 135000), (4000, 135000), (32000, 1200000)],
    ylim=[(0, 1400), (0, 1400), (0, 1200)],
    xticks=[seq_lengths, seq_lengths, extended_lengths],
    xticklabels=[
        [f"{x}" for x in seq_lengths],
        [f"{x}" for x in seq_lengths],
        [f"{x//1000}K" if x >= 1000 else f"{x}" for x in extended_lengths]
    ],
    style_name="research_paper"
)

# Save the figure
pp.save_figure(fig, "figures/9_triple_subplot", formats=["png"])

# ============================================================================
# 10. Loss Comparison Bar Chart
# ============================================================================
print("\n10. Demonstrating Loss Comparison Bar Chart...")

# Sample data for loss comparison across different models
delta_values = ['$\\delta=10^{-2}$', '$\\delta=10^{-3}$', '$\\delta=10^{-4}$']

# Data for different models and steps
step_data = {
    "10": [
        [0.349, 0.453, 0.599],  # Model 1 (LDS 8192)
        [0.118, 0.106, 0.113],  # Model 2 (LDS 1024)
        [0.0133, 0.0127, 0.013]  # Model 3 (SpectraLDS)
    ],
    "100": [
        [0.143, 0.18, 0.229],
        [0.0672, 0.0497, 0.0531],
        [0.00046, 0.000402, 0.000361]
    ],
    "2000": [
        [0.0215, 0.0253, 0.0303],
        [0.00954, 0.00734, 0.00772],
        [0.000421, 0.000359, 0.000318]
    ]
}

model_names = ["LDS 8192", "LDS 1024", "SpectraLDS"]
execution_times = {
    "LDS 8192": 2150,
    "LDS 1024": 248,
    "SpectraLDS": 42
}

# Create and display loss comparison bar chart
fig, ax = pp.loss_comparison_bar(
    delta_values=delta_values,
    step_data=step_data,
    model_names=model_names,
    title="Loss Comparison on Learning Symmetric LDSs",
    ylabel="Loss (log scale)",
    execution_times=execution_times,
    style_name="bar_chart"
)

# Save the figure
pp.save_figure(fig, "figures/10_loss_comparison_bar", formats=["png"])

# ============================================================================
# 11. Basis Functions Plot
# ============================================================================
print("\n11. Demonstrating Basis Functions Plot...")

# Generate synthetic basis functions
n_points = 8192
n_basis = 20

# Create sample basis vectors
t = np.linspace(0, 1, n_points)
basis_vectors = np.zeros((n_points, n_basis))

for i in range(n_basis):
    # Create some interesting basis functions
    freq = (i + 1) * np.pi
    phase = i * np.pi / 10
    decay = 0.5 + i/20
    
    basis_vectors[:, i] = np.sin(freq * t + phase) * np.exp(-decay * t)

# Selected indices to plot
selected_indices = [0, 2, 4, 9, 14, 19]

# Compute normalized plot data for selected indices to determine ylim
max_abs_vals = np.max(np.abs(basis_vectors), axis=0)
normalized = basis_vectors / max_abs_vals[None, :]
plot_data = normalized * 0.0002  # norm_scale default
plot_data = plot_data[:, selected_indices]

# Compute y-limits with margin
ymin = np.min(plot_data)
ymax = np.max(plot_data)
yrange = ymax - ymin
margin = 0.1 * yrange if yrange > 0 else 1e-6
ylim = (ymin - margin, ymax + margin)

# x-limits
xlim = (0, n_points)

# Create and display basis functions plot
fig, ax = pp.basis_functions_plot(
    basis_vectors=basis_vectors,
    selected_indices=selected_indices,
    n_points=n_points,
    title="Visualization of Basis Functions",
    xlabel="Position",
    ylabel="Value",
    style_name="basis_functions",
    xlim=xlim,
    ylim=ylim
)

# Save the figure
pp.save_figure(fig, "figures/11_basis_functions", formats=["png"])

# ============================================================================
# 12. Heatmap Example
# ============================================================================
print("\n12. Demonstrating Heatmap...")

# Generate a correlation matrix for features
n_features = 10

# First create random feature data
n_samples = 500
data = np.random.randn(n_samples, n_features)
# Add some correlations
data[:, 1] = data[:, 0] * 0.8 + 0.2 * data[:, 1]  # Strong positive correlation
data[:, 2] = -0.7 * data[:, 0] + 0.3 * data[:, 2]  # Strong negative correlation
data[:, 4] = data[:, 3] * 0.6 + 0.4 * data[:, 4]   # Moderate correlation

# Calculate correlation matrix
corr = np.corrcoef(data, rowvar=False)

# Feature labels
feature_names = [f"Feature {i+1}" for i in range(n_features)]

# Create and display heatmap
fig, ax = pp.heatmap(
    data=corr,
    title="Feature Correlation Matrix",
    xlabel="Features",
    ylabel="Features",
    xticklabels=feature_names,
    yticklabels=feature_names,
    colorbar_label="Correlation",
    annotate=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    style_name="research_paper"
)

# Save the figure
pp.save_figure(fig, "figures/12_heatmap", formats=["png"])

print("\nAll examples completed. Plots saved in the 'figures' directory.")