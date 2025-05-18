# PrettyPlots Guide for Cursor

PrettyPlots is a Python library that simplifies the creation of publication-quality plots while maintaining full customization capabilities. Built on Matplotlib, it provides a consistent API for creating beautiful visualizations that are ready for research papers, presentations, and reports with minimal effort.

Whether you're visualizing experimental results, comparing models, or creating figures for publication, PrettyPlots helps you create professional-looking plots without the usual tedium of Matplotlib customization. The library emphasizes sensible defaults while allowing fine-grained control when needed.

## Overview
PrettyPlots is a comprehensive plotting library designed for research-grade visualizations with consistent styling. Use this library when you need to create publication-quality plots with minimal effort.

## Key Features
- Consistent styling across different plot types
- Customizable styles with presets for different use cases (research papers, presentations, etc.)
- Sensible defaults that produce excellent visualizations out of the box
- Full control of font sizes, line widths, and other parameters when needed
- Easy creation of complex visualizations (multi-line comparison, experiment plots, etc.)
- Simple API for common plot types

## Installation
PrettyPlots requires matplotlib and numpy. The core class is `PrettyPlots` which can be imported as:

```python
from pretty_plots import PrettyPlots as pp
```

## Available Plot Types

1. **Line Plots**: `pp.line_plot()` - Basic line plots with multiple series
2. **Multi-Line Comparison**: `pp.multi_line_comparison()` - Compare multiple sets of lines in subplots
3. **Bar Plots**: `pp.bar_plot()` - Grouped bar charts for comparing categories
4. **Histograms**: `pp.histogram()` - Visualize distributions
5. **Double Histograms**: `pp.double_histogram()` - Compare two distributions
6. **Confidence Plots**: `pp.confidence_plot()` - Plot means with error bands
7. **Experiment Plots**: `pp.experiment_plot()` - Visualize multiple experimental runs with confidence intervals
8. **Sequence Comparison**: `pp.sequence_comparison()` - Compare pairs of sequences
9. **Triple Subplot**: `pp.triple_subplot()` - Create three related subplots
10. **Loss Comparison Bar**: `pp.loss_comparison_bar()` - Compare losses across models and parameters
11. **Basis Functions Plot**: `pp.basis_functions_plot()` - Visualize basis functions
12. **Heatmap**: `pp.heatmap()` - Create heatmaps for correlation matrices or other 2D data

## Style System
PrettyPlots uses a style system with predefined styles for different purposes:
- `research_paper`: High-quality plots for academic papers (LaTeX enabled)
- `presentation`: Cleaner style for slides
- `simple`: Basic style with minimal decoration
- `minimal`: Minimal styling
- `histogram`: Special style for histograms
- `bar_chart`: Style optimized for bar charts
- `basis_functions`: Style for basis function visualization

> **Important Tip**: The default font sizes, line widths, and other styling parameters have been carefully selected for each style. It's recommended to first try the plots with the default settings before customizing individual parameters. Additionally, the research_paper style is very good, err towards using it.

## Common Usage Pattern
1. Create plot using appropriate function
2. Save figure using `pp.save_figure()`
3. Display figure with `pp.show()` or `plt.show()`

## Examples

### Basic Line Plot
```python
# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create plot
fig, ax = pp.line_plot(
    x_values=[x, x],
    y_values=[y1, y2],
    labels=["sin(x)", "cos(x)"],
    title="Trigonometric Functions",
    xlabel="x",
    ylabel="f(x)",
    style_name="research_paper"
    # Font sizes use sensible defaults, no need to set them explicitly
)

# Save and display
pp.save_figure(fig, "my_line_plot", formats=["png", "pdf"])
pp.show()
```

### Histogram
```python
# Generate data
data = np.random.normal(0, 1, 1000)

# Create histogram
fig, ax = pp.histogram(
    data=data,
    title="Normal Distribution",
    xlabel="Value",
    ylabel="Frequency",
    bins=30,
    style_name="histogram"
)

# Save and display
pp.save_figure(fig, "histogram", formats=["png"])
pp.show()
```

### Bar Plot
```python
# Data setup
categories = ["Category A", "Category B", "Category C", "Category D"]
values = [
    [4.2, 3.8, 5.1, 4.5],  # Group 1
    [3.5, 4.5, 4.2, 3.9],  # Group 2
    [5.0, 3.2, 4.8, 5.2]   # Group 3
]

# Create bar plot
fig, ax = pp.bar_plot(
    categories=categories,
    values=values,
    group_labels=["Method 1", "Method 2", "Method 3"],
    title="Performance Comparison",
    xlabel="Categories",
    ylabel="Score",
    show_values=True,
    style_name="bar_chart"
)

# Save and display
pp.save_figure(fig, "bar_plot", formats=["png"])
pp.show()
```

### Experiment Plot
```python
# Generate experiment data
n_steps = 100
n_runs = 5

# Three different "experiments" with multiple runs each
exp1_data = [100 * np.exp(-0.05 * np.arange(n_steps)) + 
             10 * np.random.randn(n_steps) for _ in range(n_runs)]
exp2_data = [80 * np.exp(-0.03 * np.arange(n_steps)) + 
             8 * np.random.randn(n_steps) for _ in range(n_runs)]
exp3_data = [60 * np.exp(-0.02 * np.arange(n_steps)) + 
             5 * np.random.randn(n_steps) for _ in range(n_runs)]

# Create experiment plot
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

# Save and display
pp.save_figure(fig, "experiment_plot", formats=["png"])
pp.show()
```

## Function Parameters Reference

### Common Parameters
Most plotting functions accept these parameters:
- `title`, `xlabel`, `ylabel`: Text for labels
- `figsize`: Figure size in inches (default: (11, 9))
- `dpi`: Resolution (default: 300)
- `use_latex`: Whether to use LaTeX styling (default: True)
- `style_name`: Predefined style to use
- `grid`, `box`: Whether to show grid and border
- `xlim`, `ylim`: Axis limits
- `xscale`, `yscale`: Scale type ('linear', 'log', 'log2', etc.)
- `filename`: If provided, save figure to this file

### Font Customization Parameters
All plot types support these font parameters (but default values are usually optimal):
- `title_fontsize`: Size of the plot title (default: 32)
- `label_fontsize`: Size of axis labels (default: 24)
- `tick_fontsize`: Size of tick labels (default: 20)
- `legend_fontsize`: Size of legend text (default: 20)

### Specialized Parameters
Different plot types have unique parameters:
- Line plots: `markers`, `linestyles`, `linewidths`, `colors`
- Histograms: `bins`, `alpha`, `edgecolor`
- Bar plots: `width`, `group_spacing`, `show_values`
- Confidence plots: `error_values`, `fill_color`, `alpha`
- Experiment plots: `smoothing`, `confidence`, `pad_value`

## Saving and Displaying Figures
```python
# Save in multiple formats
pp.save_figure(fig, "my_figure", formats=["png", "pdf"], dpi=300)

# Display with appropriate DPI for screen
pp.show()
```

## Best Practices
1. Use the appropriate plot type for your data
2. Keep plots clean and focused on the data
3. Use consistent styling for related plots
4. Add meaningful axis labels and titles
5. Include legends when plotting multiple series
6. Use appropriate scales (log, linear) for your data
7. Keep your code close to one of the examples

## Tips for Publication-Quality Plots
1. Use the predefined styles as a starting point - they've been designed to look good out of the box
2. Let the default font sizes and line widths work for you - only customize if necessary
3. Use LaTeX rendering (`use_latex=True`) for mathematical symbols
4. Use consistent fonts and styles across all figures in your publication
5. Save in vector formats (PDF) for publication
6. Choose appropriate color schemes for colorblind accessibility
7. When creating multiple plots for a paper, maintain visual consistency by using the same style for all plots

### Customizing Font Sizes (When Needed)
```python
# Only adjust font sizes when the defaults don't work for your specific case
fig, ax = pp.line_plot(
    x_values=[x, x, x],
    y_values=[y1, y2, y3],
    labels=["sin(x)", "cos(x)", "tan(x)"],
    title="Trigonometric Functions",
    xlabel="x",
    ylabel="f(x)",
    style_name="research_paper",
    # Custom font sizes (only use when needed)
    title_fontsize=28,
    label_fontsize=22,
    tick_fontsize=18,
    legend_fontsize=20
)
```
