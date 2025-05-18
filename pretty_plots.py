import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.ticker as ticker
from typing import List, Dict, Optional, Union, Tuple, Any
import math

class PrettyPlots:
    """
    A comprehensive plotting library for research visualizations.
    
    This library provides a unified API for creating beautiful research plots
    with consistent styling across different plot types.
    """
    
    # Default colors with nice palette
    COLORS = [
        '#1f77b4',  # blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-green
        '#17becf',  # Cyan
        '#1a55FF',  # Light blue
    ]
    
    # Vibrant basis function colors
    BASIS_COLORS = [
        '#FF3D3D',  # Red for φ1
        '#FF8C5A',  # Coral for φ3
        '#E6D576',  # Khaki for φ5
        '#6CDBD0',  # Brighter turquoise for φ10
        '#4B71EA',  # Royal blue for φ15
        '#9A5CE6',  # Medium purple for φ20
    ]
    
    # Blues palette
    BLUES = [
        '#000080',  # Navy blue
        '#4682B4',  # Steel blue
        '#87CEEB',  # Sky blue
    ]

    @staticmethod
    def custom_styles(): #these are all basically the same rn
        """
        Return a dictionary of custom styles for different plot types.
        
        Each style is a dictionary of parameters that can be passed to the
        corresponding plotting function.
        
        Returns
        -------
        styles : dict
            Dictionary of plot styles
        """
        return {
            'research_paper': {
                'use_latex': True,
                'grid': True,
                'box': True,
                'title_fontsize': 32,
                'label_fontsize': 24,
                'tick_fontsize': 20,
                'tick_rotation': 45,
                'bold_ticks': True,
                'linewidth': 2,
            },
            'presentation': {
                'use_latex': False,
                'grid': True,
                'box': True,
                'title_fontsize': 28,
                'label_fontsize': 22,
                'tick_fontsize': 18,
                'bold_ticks': True,
                'linewidth': 2.5,
                'alpha': 0.9,
            },
            'simple': {
                'use_latex': False,
                'grid': True,
                'box': False,
                'title_fontsize': 24,
                'label_fontsize': 18,
                'tick_fontsize': 14,
                'bold_ticks': False,
                'linewidth': 1.5,
            },
            'minimal': {
                'use_latex': False,
                'grid': False,
                'box': False,
                'title_fontsize': 20,
                'label_fontsize': 16,
                'tick_fontsize': 12,
                'bold_ticks': False,
                'linewidth': 1.0,
            },
            'histogram': {
                'use_latex': True,
                'grid': True,
                'box': False,
                'bins': 30,
                'alpha': 0.85,
                'edgecolor': 'black',
                'linewidth': 0.5,
            },
            'bar_chart': {
                'use_latex': True,
                'grid': True,
                'grid_axis': 'y',
                'box': True,
                'width': 0.07,
                'group_spacing': 0.25,
                'alpha': 0.9,
                'edgecolor': 'black',
            },
            'basis_functions': {
                'use_latex': True,
                'grid': False,
                'box': True,
                'normalize': True,
                'norm_scale': 0.0002,
                'ylim': (-0.00005, 0.00005),
            },
        }

    @staticmethod
    def apply_style(params: Dict, style_name: str) -> Dict:
        """
        Apply a named style to parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary of parameters
        style_name : str
            Name of the style to apply
            
        Returns
        -------
        updated_params : dict
            Parameters with style applied
        """
        styles = PrettyPlots.custom_styles()
        
        if style_name not in styles:
            raise ValueError(f"Style '{style_name}' not found. Available styles: {list(styles.keys())}")
            
        # Create a new dictionary with style parameters
        updated_params = styles[style_name].copy()
        
        # Override with any user-provided parameters
        updated_params.update(params)
        
        return updated_params

    @staticmethod
    def save_figure(
        fig: plt.Figure,
        filename: str,
        dpi: int = 300,
        formats: List[str] = ['png', 'pdf'],
        bbox_inches: str = 'tight',
        transparent: bool = False,
        **kwargs
    ) -> None:
        """
        Save a figure in multiple formats.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        filename : str
            Base filename without extension
        dpi : int
            Resolution
        formats : list of str
            List of formats to save in
        bbox_inches : str
            Bounding box in inches
        transparent : bool
            Whether to use transparent background
        **kwargs
            Additional parameters for savefig
        """
        for fmt in formats:
            full_filename = f"{filename}.{fmt}"
            fig.savefig(full_filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent, **kwargs)
            print(f"Saved figure to {full_filename}")
    
    @staticmethod
    def _setup_latex_style():
        """Setup LaTeX styling for professional plots."""
        plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDFs
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['text.usetex'] = True  # Use LaTeX for text rendering
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
    
    @staticmethod
    def _setup_basic_style(style_name: str = None):
        """Setup basic styling without LaTeX for faster rendering. Optionally use a custom style."""
        plt.rcParams['font.family'] = 'serif'
        if style_name:
            plt.style.use(style_name)
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
    
    @staticmethod
    def _create_figure(
        figsize: Tuple[int, int] = (11, 9), 
        dpi: int = 300, 
        use_latex: bool = True,
        style_name: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a figure with appropriate styling.
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        dpi : int
            Resolution of the figure
        use_latex : bool
            Whether to use LaTeX styling
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
            
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        if use_latex:
            PrettyPlots._setup_latex_style()
        else:
            PrettyPlots._setup_basic_style(style_name)
            
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        return fig, ax
    
    @staticmethod
    def _apply_axis_styling(
        ax: plt.Axes, 
        title: str = None, 
        xlabel: str = None, 
        ylabel: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        title_pad: int = 15,
        xlabel_pad: int = 15,
        ylabel_pad: int = 15,
        grid: bool = True,
        grid_alpha: float = 0.3,
        grid_linestyle: str = '--',
        box: bool = True,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        xscale: str = None,
        yscale: str = None,
        xticks: List = None,
        xticklabels: List = None,
        yticks: List = None,
        yticklabels: List = None,
        tick_rotation: float = 0,
        bold_ticks: bool = True,
        grid_axis: str = 'both',
    ):
        """
        Apply consistent styling to the axes.
        
        Parameters
        ----------
        ax : plt.Axes
            The axes to style
        title, xlabel, ylabel : str
            Text for title and axis labels
        Various font sizes, padding, and styling options
        grid_axis : str
            Which axis to draw grid lines on ('x', 'y', or 'both')
        """
        # Set title and labels
        if title:
            ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=xlabel_pad)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=ylabel_pad)
            
        # Set scales if provided
        if xscale:
            if xscale == 'log2':
                ax.set_xscale('log', base=2)
            else:
                ax.set_xscale(xscale)
        if yscale:
            if yscale == 'log2':
                ax.set_yscale('log', base=2)
            else:
                ax.set_yscale(yscale)
            
        # Set limits if provided
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
        # Set ticks and tick labels
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, rotation=tick_rotation)
        if yticks is not None:
            ax.set_yticks(yticks)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
            
        # Set tick styling
        if bold_ticks:
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                
        # Set tick sizes
        ax.tick_params(axis='both', labelsize=tick_fontsize, length=5, width=2)
            
        # Set grid
        if grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, axis=grid_axis)
        else:
            ax.grid(False)
            
        # Set box
        if box:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            
    @staticmethod
    def _apply_legend_styling(
        ax: plt.Axes,
        fontsize: int = 20,
        framealpha: float = 0.9,
        loc: str = 'upper right'
    ):
        """Apply consistent legend styling."""
        if ax.get_legend():
            ax.get_legend().remove()  # Remove existing legend
            
        ax.legend(fontsize=fontsize, framealpha=framealpha, loc=loc)
        
    @staticmethod
    def line_plot(
        x_values: List[List[Union[int, float]]],
        y_values: List[List[Union[int, float]]],
        labels: List[str] = None,
        title: str = "Line Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        figsize: Tuple[int, int] = (11, 9),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        colors: List[str] = None,
        markers: List[str] = None,
        linestyles: List[str] = None,
        linewidths: List[float] = None,
        xscale: str = None,
        yscale: str = None,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        grid: bool = True,
        box: bool = True,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a line plot.
        
        Parameters
        ----------
        x_values : list of arrays
            List of x values for each line
        y_values : list of arrays
            List of y values for each line
        labels : list of str
            Labels for each line
        title, xlabel, ylabel : str
            Text for title and axis labels
        figsize : tuple
            Figure size in inches
        dpi : int
            Resolution of the figure
        use_latex : bool
            Whether to use LaTeX styling
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        colors, markers, linestyles, linewidths : list
            Styling for each line
        xscale, yscale : str
            Scale for the axes
        xlim, ylim : tuple
            Limits for the axes
        grid, box : bool
            Whether to show grid and box
        filename : str
            If provided, save the figure to this file
        **kwargs
            Additional styling parameters
            
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        fig, ax = PrettyPlots._create_figure(figsize, dpi, use_latex, style_name)
        
        # Use default colors if none provided
        if colors is None:
            colors = PrettyPlots.COLORS
            
        # Handle single line case (not in a list)
        if not isinstance(x_values[0], (list, np.ndarray)):
            x_values = [x_values]
        if not isinstance(y_values[0], (list, np.ndarray)):
            y_values = [y_values]
            
        # Make sure we have enough values for each line
        n_lines = min(len(x_values), len(y_values))
        if labels is None:
            labels = [f"Line {i+1}" for i in range(n_lines)]
        else:
            labels = labels[:n_lines]
            
        # Set default values for markers, linestyles, and linewidths
        if markers is None:
            markers = ['o'] * n_lines
        elif len(markers) < n_lines:
            markers.extend(['o'] * (n_lines - len(markers)))
            
        if linestyles is None:
            linestyles = ['-'] * n_lines
        elif len(linestyles) < n_lines:
            linestyles.extend(['-'] * (n_lines - len(linestyles)))
            
        if linewidths is None:
            linewidths = [2] * n_lines
        elif len(linewidths) < n_lines:
            linewidths.extend([2] * (n_lines - len(linewidths)))
        
        # Plot each line
        for i in range(n_lines):
            ax.plot(
                x_values[i], 
                y_values[i], 
                label=labels[i], 
                color=colors[i % len(colors)],
                marker=markers[i],
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                alpha=0.8
            )
            
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            xlabel=xlabel, 
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            box=box,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        # Add legend
        PrettyPlots._apply_legend_styling(ax, fontsize=legend_fontsize, **kwargs)
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, ax
        
    @staticmethod
    def multi_line_comparison(
        x_values: List[List[Union[int, float]]],
        y_values: List[List[Union[int, float]]],
        subplot_titles: List[str],
        labels: List[str] = None,
        main_title: str = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        figsize: Tuple[int, int] = (20, 9),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        colors: List[str] = None,
        markers: List[str] = None,
        linestyles: List[str] = None,
        linewidths: List[float] = None,
        xscale: str = None,
        yscale: str = None,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        grid: bool = True,
        box: bool = True,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create a multi-subplot line comparison chart.
        
        Parameters
        ----------
        Similar to line_plot, but with multiple subplots
        subplot_titles : list of str
            Titles for each subplot
        main_title : str
            Overall title for the figure
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
            
        Returns
        -------
        fig, axes : tuple
            Figure and list of axes objects
        """
        if colors is None:
            colors = PrettyPlots.COLORS
            
        # Determine number of subplots
        n_plots = len(subplot_titles)
        
        if use_latex:
            PrettyPlots._setup_latex_style()
        else:
            PrettyPlots._setup_basic_style(style_name)
            
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, dpi=dpi)
        
        # Handle case of single subplot
        if n_plots == 1:
            axes = [axes]
            
        # Plot on each subplot
        for i, ax in enumerate(axes):
            # Get data for this subplot
            curr_x_values = x_values[i] if i < len(x_values) else x_values[0]
            curr_y_values = y_values[i] if i < len(y_values) else y_values[0]
            
            # Handle single line case
            if not isinstance(curr_x_values[0], (list, np.ndarray)):
                curr_x_values = [curr_x_values]
            if not isinstance(curr_y_values[0], (list, np.ndarray)):
                curr_y_values = [curr_y_values]
                
            # Make sure we have enough values for each line
            n_lines = min(len(curr_x_values), len(curr_y_values))
            curr_labels = labels[:n_lines] if labels is not None else [f"Line {j+1}" for j in range(n_lines)]
            
            # Set default values for markers, linestyles, and linewidths
            curr_markers = markers[:n_lines] if markers is not None else ['o'] * n_lines
            curr_linestyles = linestyles[:n_lines] if linestyles is not None else ['-'] * n_lines
            curr_linewidths = linewidths[:n_lines] if linewidths is not None else [2] * n_lines
            
            # Plot each line
            for j in range(n_lines):
                ax.plot(
                    curr_x_values[j], 
                    curr_y_values[j], 
                    label=curr_labels[j], 
                    color=colors[j % len(colors)],
                    marker=curr_markers[j],
                    linestyle=curr_linestyles[j],
                    linewidth=curr_linewidths[j],
                    alpha=0.8
                )
                
            # Apply styling
            PrettyPlots._apply_axis_styling(
                ax, 
                title=subplot_titles[i], 
                xlabel=xlabel if i == 0 or n_plots <= 2 else "",
                ylabel=ylabel if i == 0 else "",
                xscale=xscale,
                yscale=yscale,
                xlim=xlim,
                ylim=ylim,
                grid=grid,
                box=box,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                **kwargs
            )
            
            # Add legend to the first subplot or all if specified
            if i == 0 or kwargs.get('legend_all', False):
                PrettyPlots._apply_legend_styling(ax, fontsize=legend_fontsize, **kwargs)
                
        # Add overall title if provided
        if main_title:
            fig.suptitle(main_title, fontsize=title_fontsize, y=1.05)
            
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, axes
    
    @staticmethod
    def bar_plot(
        categories: List[str],
        values: List[List[float]],
        group_labels: List[str] = None,
        bar_labels: List[str] = None,
        title: str = "Bar Plot",
        xlabel: str = None,
        ylabel: str = "Value",
        figsize: Tuple[int, int] = (11, 9),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        colors: List[str] = None,
        yscale: str = None,
        ylim: Tuple[float, float] = None,
        grid: bool = True,
        box: bool = True,
        width: float = 0.2,
        group_spacing: float = 0.25,
        show_values: bool = False,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a grouped bar plot.
        
        Parameters
        ----------
        categories : list of str
            Names of categories (x-axis)
        values : list of lists
            Values for each group and category
        group_labels : list of str
            Labels for each group
        bar_labels : list of str
            Labels to display below the bars (e.g., step sizes)
        title, xlabel, ylabel : str
            Text for title and axis labels
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        Other styling parameters similar to line_plot
        width : float
            Width of each bar
        group_spacing : float
            Spacing between groups
        show_values : bool
            Whether to show values above bars
            
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        fig, ax = PrettyPlots._create_figure(figsize, dpi, use_latex, style_name)
        
        # Use default colors if none provided
        if colors is None:
            colors = PrettyPlots.BLUES
            
        # Determine number of groups and categories
        n_groups = len(values)
        n_categories = len(categories)
        
        # Set default group labels if not provided
        if group_labels is None:
            group_labels = [f"Group {i+1}" for i in range(n_groups)]
            
        # Calculate positions for bars
        positions = np.arange(n_categories)
        
        # Create positions for each group of bars
        group_positions = []
        for i in range(n_groups):
            offset = (i - (n_groups - 1) / 2) * width
            group_positions.append(positions + offset)
            
        # Plot bars for each group
        bars = []
        for i in range(n_groups):
            bars.append(
                ax.bar(
                    group_positions[i],
                    values[i],
                    width,
                    label=group_labels[i],
                    color=colors[i % len(colors)],
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )
            )
            
        # Show values above bars if requested
        if show_values:
            for i in range(n_groups):
                for j, bar in enumerate(bars[i]):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height * 1.01,
                        f"{height}",
                        ha='center',
                        va='bottom',
                        fontsize=20,
                        rotation=90
                    )
                    
        # Add custom bar labels if provided
        if bar_labels is not None:
            # Add bar labels below the categories
            for i, pos in enumerate(positions):
                if i < len(bar_labels):
                    ax.text(
                        pos, 
                        ax.get_ylim()[0] * 1.05, 
                        bar_labels[i],
                        ha='center',
                        va='top',
                        fontsize=20,
                        fontweight='bold'
                    )
                    
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            xlabel=xlabel,
            ylabel=ylabel,
            yscale=yscale,
            ylim=ylim,
            grid=grid,
            box=box,
            xticks=positions,
            xticklabels=categories,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        # Add legend
        PrettyPlots._apply_legend_styling(ax, fontsize=legend_fontsize, **kwargs)
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, ax
        
    @staticmethod
    def histogram(
        data: Union[List[float], np.ndarray],
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        figsize: Tuple[int, int] = (11, 9),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        color: str = '#4C72B0',
        bins: int = 30,
        alpha: float = 0.85,
        edgecolor: str = 'black',
        grid: bool = True,
        box: bool = False,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a histogram.
        
        Parameters
        ----------
        data : array-like
            Data to plot
        title, xlabel, ylabel : str
            Text for title and axis labels
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        Other styling parameters similar to line_plot
        bins : int
            Number of bins
        alpha : float
            Transparency of the bars
        edgecolor : str
            Color of the bar edges
            
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        fig, ax = PrettyPlots._create_figure(figsize, dpi, use_latex, style_name)
        
        # Plot histogram
        ax.hist(
            data,
            bins=bins,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=0.5
        )
        
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            box=box,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, ax
        
    @staticmethod
    def double_histogram(
        data1: Union[List[float], np.ndarray],
        data2: Union[List[float], np.ndarray],
        label1: str = "Data 1",
        label2: str = "Data 2",
        title: str = "Histogram Comparison",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        figsize: Tuple[int, int] = (11, 9),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        color1: str = '#FF3D3D',
        color2: str = '#4B71EA',
        bins: int = 30,
        alpha: float = 0.7,
        edgecolor: str = 'black',
        grid: bool = True,
        box: bool = True,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a histogram with two datasets.
        
        Parameters
        ----------
        data1, data2 : array-like
            Data to plot
        label1, label2 : str
            Labels for the datasets
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        Other parameters similar to histogram()
            
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        fig, ax = PrettyPlots._create_figure(figsize, dpi, use_latex, style_name)
        
        # Plot histograms
        ax.hist(
            data1,
            bins=bins,
            color=color1,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=0.5,
            label=label1
        )
        
        ax.hist(
            data2,
            bins=bins,
            color=color2,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=0.5,
            label=label2
        )
        
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            box=box,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        # Add legend
        PrettyPlots._apply_legend_styling(ax, fontsize=legend_fontsize, **kwargs)
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, ax
    
    @staticmethod
    def basis_functions_plot(
        basis_vectors: Union[List[List[float]], np.ndarray, 'torch.Tensor'],
        selected_indices: List[int] = None,
        n_points: int = 8192,
        title: str = "Basis Functions",
        xlabel: str = "Position",
        ylabel: str = "Value",
        figsize: Tuple[int, int] = (11, 9),
        dpi: int = 300,
        colors: List[str] = None,
        ylim: Tuple[float, float] = None,
        legend_labels: List[str] = None,
        normalize: bool = True,
        norm_scale: float = 0.0002,
        grid: bool = False,
        box: bool = True,
        filename: str = None,
        style_name: str = None,
        xlim: Tuple[float, float] = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a plot of basis functions.
        
        Parameters
        ----------
        basis_vectors : array-like
            Matrix of basis vectors (shape [n_points, n_vectors])
        selected_indices : list of int
            Indices of basis vectors to plot
        n_points : int
            Number of points to plot
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        Other parameters similar to line_plot
        normalize : bool
            Whether to normalize the basis vectors (always True for user-friendliness)
        norm_scale : float
            Scale factor for normalization
        xlim, ylim : tuple
            Limits for the axes (auto-computed if not provided)
        
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        # Convert to numpy if needed
        if 'torch' in str(type(basis_vectors)):
            # Assuming it's a PyTorch tensor
            import torch
            if torch.is_tensor(basis_vectors):
                basis_vectors = basis_vectors.detach().cpu().numpy()
            else:
                basis_vectors = np.array(basis_vectors)
        elif not isinstance(basis_vectors, np.ndarray):
            basis_vectors = np.array(basis_vectors)

        # Select all columns if not specified
        if selected_indices is None:
            selected_indices = list(range(min(6, basis_vectors.shape[1])))
        
        # Use default colors if none provided
        if colors is None:
            colors = PrettyPlots.BASIS_COLORS
        
        # Always normalize and scale the selected basis vectors
        max_abs_vals = np.max(np.abs(basis_vectors), axis=0)
        normalized = basis_vectors / max_abs_vals[None, :]
        plot_data = normalized * norm_scale
        plot_data = plot_data[:n_points, :] if plot_data.shape[0] > n_points else plot_data
        plot_data_sel = plot_data[:, selected_indices]
        
        # Create x values for plotting
        x = np.linspace(0, n_points, min(n_points, plot_data.shape[0]))
        
        # Auto-compute ylim if not provided
        if ylim is None:
            ymin = np.min(plot_data_sel)
            ymax = np.max(plot_data_sel)
            yrange = ymax - ymin
            margin = 0.1 * yrange if yrange > 0 else 1e-6
            ylim = (ymin - margin, ymax + margin)
        # Auto-compute xlim if not provided
        if xlim is None:
            xlim = (0, n_points)
        
        # Create the figure
        fig, ax = PrettyPlots._create_figure(figsize, dpi, True, style_name)  # Always use LaTeX for this plot
        
        # Plot the selected basis functions
        for i, idx in enumerate(selected_indices):
            if idx < plot_data.shape[1]:
                ax.plot(
                    x, 
                    plot_data[:, idx], 
                    color=colors[i % len(colors)], 
                    linewidth=2, 
                    antialiased=True
                )
        
        # Create default legend labels if not provided
        if legend_labels is None:
            legend_labels = [f'$\\varphi_{{{idx+1}}}$' for idx in selected_indices]
        
        # Remove xlim/ylim from kwargs if present to avoid duplicate arguments
        kwargs.pop('xlim', None)
        kwargs.pop('ylim', None)
        
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            xlabel=xlabel,
            ylabel=ylabel,
            ylim=ylim,
            grid=grid,
            box=box,
            xlim=xlim,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        # Add legend with custom labels
        ax.legend(legend_labels, fontsize=legend_fontsize, framealpha=0.9, loc='upper right')
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def confidence_plot(
        x_values: Union[List[float], np.ndarray],
        y_values: Union[List[float], np.ndarray],
        error_values: Union[List[float], np.ndarray],
        title: str = "Confidence Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        figsize: Tuple[int, int] = (11, 9),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        color: str = '#1f77b4',
        fill_color: str = None,
        line_width: float = 2.5,
        alpha: float = 0.2,
        xscale: str = None,
        yscale: str = None,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        grid: bool = True,
        box: bool = True,
        label: str = "Mean Value",
        confidence_label: str = "Confidence Interval",
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a line plot with confidence interval.
        
        Parameters
        ----------
        x_values : array-like
            X values for the plot
        y_values : array-like
            Y values for the plot (mean values)
        error_values : array-like
            Error values (standard deviation or similar)
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        Other parameters similar to line_plot
        fill_color : str
            Color for the confidence interval (if None, uses the line color)
        alpha : float
            Transparency of the confidence interval
            
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        fig, ax = PrettyPlots._create_figure(figsize, dpi, use_latex, style_name)
        
        # Set fill color to match line color if not specified
        if fill_color is None:
            fill_color = color
            
        # Plot mean line
        ax.plot(
            x_values, 
            y_values, 
            color=color, 
            linewidth=line_width, 
            label=label
        )
        
        # Plot confidence interval
        ax.fill_between(
            x_values, 
            y_values - error_values, 
            y_values + error_values, 
            color=fill_color, 
            alpha=alpha,
            label=confidence_label
        )
        
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            box=box,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        # Add legend
        PrettyPlots._apply_legend_styling(ax, fontsize=legend_fontsize, **kwargs)
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, ax
    
    @staticmethod
    def experiment_plot(
        data_dict: Dict[str, List[List[float]]],
        title: str = "Experiment Results",
        xlabel: str = "Step",
        ylabel: str = "Value",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        pad_value: float = None,
        smoothing: int = None,
        confidence: float = 1.96,  # 95% confidence interval
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        grid: bool = True,
        box: bool = False,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a plot for experiment results with confidence intervals.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary mapping experiment names to lists of result arrays
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        Other parameters similar to line_plot
        pad_value : float
            Value to pad shorter arrays with
        smoothing : int
            Window size for smoothing
        confidence : float
            Multiplier for error bounds (1.96 = 95% confidence)
            
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        fig, ax = PrettyPlots._create_figure(figsize, dpi, use_latex, style_name)
        
        # Determine maximum length of all arrays
        max_len = 0
        for values in data_dict.values():
            max_len = max(max_len, max(len(v) for v in values))
            
        # Process each experiment
        for key, values_list in data_dict.items():
            # Pad arrays if requested
            if pad_value is not None:
                padded_values = []
                for values in values_list:
                    old_len = len(values)
                    if old_len < max_len:
                        padded = np.pad(values, (0, max_len - old_len), 'constant', constant_values=pad_value)
                    else:
                        padded = np.array(values)
                    padded_values.append(padded)
                values_array = np.stack(padded_values)
            else:
                # Stack without padding (assumes all arrays are same length)
                values_array = np.stack([np.array(v) for v in values_list])
                
            # Create x-axis values
            x = np.arange(values_array.shape[1])
            
            # Calculate mean and standard deviation
            mean = np.nanmean(values_array, axis=0)
            std = np.nanstd(values_array, axis=0)
            
            # Apply smoothing if requested
            if smoothing is not None and smoothing > 1:
                # Create smoothing window
                window = np.ones(smoothing) / smoothing
                
                # Apply convolution to smooth mean and variance
                mean = np.convolve(mean, window, mode='valid')
                std_smoothed = np.sqrt(np.convolve(std**2, window**2, mode='valid'))
                
                # Adjust x values for smoothed data
                x = x[smoothing-1:smoothing-1+len(mean)]
                std = std_smoothed
                
            # Plot mean line
            ax.plot(x, mean, label=key, linewidth=2)
            
            # Plot confidence interval if multiple runs
            if len(values_list) > 1:
                ax.fill_between(
                    x, 
                    mean - confidence * std, 
                    mean + confidence * std,
                    alpha=0.2
                )
                
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            box=box,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        # Add legend
        PrettyPlots._apply_legend_styling(ax, fontsize=legend_fontsize, **kwargs)
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, ax
    
    @staticmethod
    def sequence_comparison(
        sequences: List[np.ndarray],
        labels: List[str] = None,
        title: str = "Sequence Comparison",
        xlabel: str = "Time",
        ylabel: str = "Value",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        colors: List[str] = None,
        linestyles: List[str] = None,
        mse_label: bool = True,
        layout: str = 'vertical',  # 'vertical', 'horizontal', or 'grid'
        n_cols: int = 3,
        grid: bool = True,
        box: bool = True,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create comparison plots for pairs of sequences.
        
        Parameters
        ----------
        sequences : list of arrays
            List of sequences to compare (should have even length)
        labels : list of str
            Labels for each sequence
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        Other parameters similar to line_plot
        mse_label : bool
            Whether to display MSE values on the plot
        layout : str
            Layout type ('vertical', 'horizontal', or 'grid')
        n_cols : int
            Number of columns for grid layout
            
        Returns
        -------
        fig, axes : tuple
            Figure and list of axes objects
        """
        # Validate sequences
        if len(sequences) % 2 != 0:
            raise ValueError("Number of sequences must be even for pairwise comparison")
            
        n_pairs = len(sequences) // 2
        
        # Set default labels if not provided
        if labels is None:
            labels = []
            for i in range(n_pairs):
                labels.extend([f"Sequence {2*i+1}", f"Sequence {2*i+2}"])
                
        # Default colors if not provided
        if colors is None:
            # Use blue and red for each pair
            colors = []
            for _ in range(n_pairs):
                colors.extend(['blue', 'red'])
                
        # Default linestyles if not provided
        if linestyles is None:
            # Use solid and dashed for each pair
            linestyles = []
            for _ in range(n_pairs):
                linestyles.extend(['-', '--'])
                
        # Determine layout
        if layout == 'vertical':
            fig, axes = plt.subplots(n_pairs, 1, figsize=figsize, dpi=dpi)
        elif layout == 'horizontal':
            fig, axes = plt.subplots(1, n_pairs, figsize=figsize, dpi=dpi)
        else:  # grid
            n_rows = math.ceil(n_pairs / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
            
        # Ensure axes is iterable
        if n_pairs == 1:
            axes = [axes]
        elif layout == 'grid':
            axes = axes.flatten()
            
        # Set LaTeX style if requested
        if use_latex:
            PrettyPlots._setup_latex_style()
        else:
            PrettyPlots._setup_basic_style(style_name)
            
        # Plot each pair
        for i in range(n_pairs):
            ax = axes[i]
            
            # Get sequences for this pair
            seq1 = sequences[2*i]
            seq2 = sequences[2*i+1]
            
            # Create x values assuming sequences are 1D
            x = np.arange(len(seq1))
            
            # Plot sequences
            ax.plot(x, seq1, label=labels[2*i], color=colors[2*i], linestyle=linestyles[2*i], linewidth=2)
            ax.plot(x, seq2, label=labels[2*i+1], color=colors[2*i+1], linestyle=linestyles[2*i+1], linewidth=2)
            
            # Calculate MSE
            mse = np.mean((seq1 - seq2)**2)
            
            # Add MSE label if requested
            if mse_label:
                ax.text(
                    0.02, 0.95, 
                    f'MSE: {mse:.8f}', 
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.7)
                )
                
            # Set title for each subplot
            ax.set_title(f'Comparison {i+1}')
            
            # Add legend
            ax.legend()
            
            # Apply styling
            PrettyPlots._apply_axis_styling(
                ax, 
                xlabel=xlabel if i == n_pairs-1 else "",
                ylabel=ylabel,
                grid=grid,
                box=box,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                **kwargs
            )
            
        # Handle empty subplots in grid layout
        if layout == 'grid':
            for i in range(n_pairs, len(axes)):
                axes[i].axis('off')
            
        # Add overall title
        if title:
            fig.suptitle(title, fontsize=title_fontsize, y=1.02)
            
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, axes
    
    @staticmethod
    def triple_subplot(
        x_values: List[List[Union[int, float]]],
        y_values: List[List[Union[int, float]]],
        subplot_titles: List[str],
        labels: List[List[str]],
        main_title: str = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        figsize: Tuple[int, int] = (30, 9),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        colors: List[str] = None,
        markers: List[str] = None,
        linestyles: List[str] = None,
        linewidths: List[float] = None,
        xscale: str = 'log2',
        yscale: str = None,
        xlim: List[Tuple[float, float]] = None,
        ylim: List[Tuple[float, float]] = None,
        xticks: List[List] = None,
        xticklabels: List[List[str]] = None,
        grid: bool = True,
        box: bool = True,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create a figure with three subplots, each with its own dataset.
        
        Parameters
        ----------
        Similar to multi_line_comparison, but allows different data for each subplot
        and separate styling for each
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
            
        Returns
        -------
        fig, axes : tuple
            Figure and list of axes objects
        """
        if colors is None:
            colors = PrettyPlots.COLORS
            
        # Set LaTeX style if requested
        if use_latex:
            PrettyPlots._setup_latex_style()
        else:
            PrettyPlots._setup_basic_style(style_name)
            
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        
        # Plot each subplot with its own data
        for i, ax in enumerate(axes):
            # Get data for this subplot
            if i < len(x_values) and i < len(y_values):
                curr_x_values = x_values[i]
                curr_y_values = y_values[i]
                curr_labels = labels[i] if i < len(labels) else None
                
                # Handle case where data is a list of arrays
                if isinstance(curr_x_values[0], (list, np.ndarray)):
                    for j, (x, y) in enumerate(zip(curr_x_values, curr_y_values)):
                        label = curr_labels[j] if curr_labels and j < len(curr_labels) else f"Line {j+1}"
                        color = colors[j % len(colors)] if colors else None
                        marker = markers[j] if markers and j < len(markers) else 'o'
                        linestyle = linestyles[j] if linestyles and j < len(linestyles) else '-'
                        linewidth = linewidths[j] if linewidths and j < len(linewidths) else 2
                        
                        ax.plot(
                            x, y, 
                            label=label, 
                            color=color,
                            marker=marker,
                            linestyle=linestyle,
                            linewidth=linewidth
                        )
                else:
                    # Single line case
                    ax.plot(
                        curr_x_values, 
                        curr_y_values, 
                        label=curr_labels[0] if curr_labels else "Line",
                        color=colors[0] if colors else None,
                        marker=markers[0] if markers else 'o',
                        linestyle=linestyles[0] if linestyles else '-',
                        linewidth=linewidths[0] if linewidths else 2
                    )
            
            # Apply styling specific to this subplot
            curr_xlabel = xlabel
            curr_ylabel = ylabel if i == 0 else ""
            curr_xlim = xlim[i] if xlim and i < len(xlim) else None
            curr_ylim = ylim[i] if ylim and i < len(ylim) else None
            curr_xticks = xticks[i] if xticks and i < len(xticks) else None
            curr_xticklabels = xticklabels[i] if xticklabels and i < len(xticklabels) else None
            
            PrettyPlots._apply_axis_styling(
                ax, 
                title=subplot_titles[i] if i < len(subplot_titles) else f"Subplot {i+1}",
                xlabel=curr_xlabel,
                ylabel=curr_ylabel,
                xscale=xscale,
                yscale=yscale,
                xlim=curr_xlim,
                ylim=curr_ylim,
                xticks=curr_xticks,
                xticklabels=curr_xticklabels,
                tick_rotation=45 if curr_xticklabels else 0,
                grid=grid,
                box=box,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                **kwargs
            )
            
            # Add legend
            PrettyPlots._apply_legend_styling(ax, fontsize=legend_fontsize, **kwargs)
            
        # Add overall title if provided
        if main_title:
            fig.suptitle(main_title, fontsize=title_fontsize, y=1.05)
            
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, axes

    @staticmethod
    def loss_comparison_bar(
        delta_values: List[str],
        step_data: Dict[str, List[List[float]]],
        model_names: List[str],
        title: str = "Loss Comparison",
        ylabel: str = "Loss (log scale)",
        step_labels: List[str] = None,
        figsize: Tuple[int, int] = (11, 9),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        colors: Dict[str, List[str]] = None,
        group_width: float = 0.25,
        bar_width: float = 0.07,
        yscale: str = 'log',
        ylim: Tuple[float, float] = (1e-4, 1),
        execution_times: Dict[str, float] = None,
        ylower_labels: Dict[str, List[str]] = None,
        filename: str = None,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        legend_fontsize: int = 20,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a grouped bar chart comparing losses for different models.
        
        Parameters
        ----------
        delta_values : list of str
            Labels for delta values
        step_data : dict
            Dictionary mapping step names to lists of loss values for each model
        model_names : list of str
            Names of the models
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        step_labels : list of str
            Labels for training steps
        execution_times : dict
            Dictionary mapping model names to execution times (optional)
        ylower_labels : dict
            Dictionary mapping label row name to list of values (each list should have length equal to number of bar groups)
        Other parameters similar to bar_plot
        
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        fig, ax = PrettyPlots._create_figure(figsize, dpi, use_latex, style_name)
        
        # Default colors if not provided
        if colors is None:
            colors = {
                model_names[0]: [PrettyPlots.BLUES[0]] * 3,
                model_names[1]: [PrettyPlots.BLUES[1]] * 3,
                model_names[2]: [PrettyPlots.BLUES[2]] * 3
            }
        
        # Default step labels if not provided
        if step_labels is None:
            step_labels = list(step_data.keys())
        
        # Positions for the bars
        positions = np.arange(len(delta_values))
        
        # Calculate positions for each step
        step_positions = {}
        for i, step in enumerate(step_data.keys()):
            step_positions[step] = positions + (i - 1) * group_width
        
        # Plot bars for each step and model
        all_bars = []
        for i, (step, values) in enumerate(step_data.items()):
            for j, model in enumerate(model_names):
                bars = ax.bar(
                    step_positions[step] - (len(model_names) // 2 - j) * bar_width,
                    values[j],
                    bar_width,
                    color=colors[model][i % len(colors[model])],
                    label=model if i == 0 else None  # Only add to legend once
                )
                all_bars.append(bars)
        
        # Set y-scale
        ax.set_yscale(yscale)
        
        # Remove x-ticks and label manually
        ax.set_xticks([])
        
        # --- Generalized lower labels ---
        # Compose all label rows: each is a (row_label, values) pair
        label_rows = []
        # Add step and delta as default rows if not overridden
        if ylower_labels is None:
            ylower_labels = {
                'Steps': list(step_data.keys()),
                'Delta': delta_values
            }
        else:
            # If user omits 'Steps' or 'Delta', add them as fallback
            if 'Steps' not in ylower_labels:
                ylower_labels = {'Steps': list(step_data.keys()), **ylower_labels}
            if 'Delta' not in ylower_labels:
                ylower_labels = {**ylower_labels, 'Delta': delta_values}
        for row_label, values in ylower_labels.items():
            label_rows.append((row_label, values))
        
        # Render all label rows below the x-axis using axes coordinates
        # Start at y=-0.08 and stack downward with offset
        base_y = -0.08
        row_offset = -0.06
        for row_idx, (row_label, values) in enumerate(label_rows):
            y = base_y + row_offset * row_idx
            # Row label (left of axis)
            ax.text(-0.06, y, row_label, ha='right', va='center', fontsize=20, color='black', fontweight='bold', transform=ax.transAxes)
            # Row values (centered under each bar group)
            for i, pos in enumerate(positions):
                val = values[i] if i < len(values) else ''
                ax.text(
                    pos, y, val,
                    ha='center', va='center', fontsize=20, color='black', fontweight='bold',
                    transform=ax.get_xaxis_transform()
                )
        
        # Add execution time summary if provided
        if execution_times:
            exec_text = "Execution Times: " + " — ".join([f"{model} ({time}s)" for model, time in execution_times.items()])
            fig.text(0.5, 0.02, exec_text, ha='center', fontsize=16, fontstyle='italic')
        
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            ylabel=ylabel,
            yscale=yscale,
            ylim=ylim,
            grid=True,
            grid_axis='y',
            box=True,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        # Add legend
        PrettyPlots._apply_legend_styling(ax, fontsize=legend_fontsize, **kwargs)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for execution time text
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        
        return fig, ax
        
    @staticmethod
    def heatmap(
        data: Union[List[List[float]], np.ndarray],
        title: str = "Heatmap",
        xlabel: str = "X",
        ylabel: str = "Y",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300,
        use_latex: bool = True,
        style_name: str = None,
        cmap: str = 'viridis',
        colorbar_label: str = "Value",
        xticklabels: List[str] = None,
        yticklabels: List[str] = None,
        annotate: bool = False,
        fmt: str = ".2f",
        vmin: float = None,
        vmax: float = None,
        filename: str = None,
        show_xlabels: bool = True,
        title_fontsize: int = 32,
        label_fontsize: int = 24,
        tick_fontsize: int = 20,
        colorbar_fontsize: int = 16,
        annotation_fontsize: int = 8,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a heatmap.
        
        Parameters
        ----------
        data : array-like
            2D array of values
        style_name : str
            Name of the matplotlib style to use (if not using LaTeX)
        Other parameters similar to other plots
        cmap : str
            Colormap to use
        colorbar_label : str
            Label for the colorbar
        annotate : bool
            Whether to annotate each cell with its value
        fmt : str
            Format string for annotations
        show_xlabels : bool
            Whether to show x-axis tick labels (default True)
        
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        fig, ax = PrettyPlots._create_figure(figsize, dpi, use_latex, style_name)
        
        # Convert data to numpy array if not already
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        # Create the heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, fontsize=colorbar_fontsize, labelpad=10)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        
        # Set ticks and labels
        if xticklabels is not None:
            ax.set_xticks(np.arange(len(xticklabels)))
            if show_xlabels:
                ax.set_xticklabels(xticklabels, rotation=90, ha='center')
            else:
                ax.set_xticklabels([])
        else:
            ax.set_xticks(np.arange(data.shape[1]))
            if not show_xlabels:
                ax.set_xticklabels([])
        
        if yticklabels is not None:
            ax.set_yticks(np.arange(len(yticklabels)))
            ax.set_yticklabels(yticklabels)
        else:
            ax.set_yticks(np.arange(data.shape[0]))
            
        # Annotate cells if requested
        if annotate:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text_color = 'white' if im.norm(data[i, j]) > 0.5 else 'black'
                    ax.text(
                        j, i, 
                        format(data[i, j], fmt), 
                        ha='center', 
                        va='center', 
                        color=text_color,
                        fontsize=annotation_fontsize
                    )
                    
        # Apply styling
        PrettyPlots._apply_axis_styling(
            ax, 
            title=title, 
            xlabel=xlabel,
            ylabel=ylabel,
            grid=False,
            box=True,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs
        )
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            
        return fig, ax

    @staticmethod
    def show():
        """
        Display the current matplotlib figure, ensuring the DPI is not too high for interactive display.
        If the current figure's DPI is above 150, it will be set to 100 for better on-screen rendering.
        The DPI will be restored to its original value after display.
        """
        fig = plt.gcf()
        orig_dpi = fig.get_dpi()
        changed = False
        if orig_dpi > 150:
            fig.set_dpi(100)
            changed = True
        plt.show()
        if changed:
            fig.set_dpi(orig_dpi)