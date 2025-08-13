## fully written by Claude Opus 4.1 (extended thinking)

"""
3D Vector Visualization Module
A reusable module for visualizing vectors in 3D space from the origin.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch


def plot_3d_vectors(vectors, 
                    colors=None, 
                    labels=None, 
                    title='3D Vector Visualization',
                    figsize=(12, 9),
                    view_elev=25,
                    view_azim=70,
                    max_range=None,
                    show_projections=True,
                    show_axes=True,
                    show_plane=True,
                    show_grid=True,
                    return_fig=False):
    """
    Plot 3D vectors from the origin with customizable visualization options.
    
    Parameters:
    -----------
    vectors : torch.Tensor, np.ndarray, or list
        Matrix where each row is a 3D vector to plot
    colors : list, optional
        List of colors for each vector. Defaults to ['red', 'green', 'blue', ...]
    labels : list, optional
        List of labels for each vector. Auto-generated if not provided
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    view_elev : float
        Elevation angle for viewing perspective
    view_azim : float
        Azimuth angle for viewing perspective
    max_range : float, optional
        Maximum range for axes. Auto-calculated if not provided
    show_projections : bool
        Whether to show projection lines to coordinate planes
    show_axes : bool
        Whether to show coordinate axes through origin
    show_plane : bool
        Whether to show the z=0 plane
    show_grid : bool
        Whether to show gridlines
    return_fig : bool
        If True, returns (fig, ax) tuple instead of showing plot
    
    Returns:
    --------
    (fig, ax) if return_fig=True, otherwise None
    """
    
    # Convert input to numpy array
    if torch.is_tensor(vectors):
        vectors = vectors.detach().numpy()
    elif isinstance(vectors, list):
        vectors = np.array(vectors)
    
    # Validate input shape
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.shape[1] != 3:
        raise ValueError("Vectors must be 3-dimensional (shape: [n, 3])")
    
    n_vectors = vectors.shape[0]
    
    # Set default colors if not provided
    if colors is None:
        default_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = default_colors[:n_vectors] if n_vectors <= len(default_colors) else default_colors * (n_vectors // len(default_colors) + 1)
        colors = colors[:n_vectors]
    
    # Set default labels if not provided
    if labels is None:
        labels = [f'Vector {i+1}: [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]' for i, v in enumerate(vectors)]
    
    # Calculate max_range if not provided
    if max_range is None:
        max_coord = np.abs(vectors).max()
        max_range = max(2.5, max_coord * 1.3)  # Add some padding
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Origin point
    origin = [0, 0, 0]
    
    # Plot coordinate axes through origin if requested
    if show_axes:
        axis_length = max_range
        axis_alpha = 0.6
        axis_width = 1.5
        
        # X-axis (red-ish)
        ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', alpha=axis_alpha, linewidth=axis_width)
        ax.plot([-axis_length, 0], [0, 0], [0, 0], 'r--', alpha=axis_alpha/2, linewidth=axis_width)
        ax.text(axis_length, 0, 0, 'X', fontsize=14, fontweight='bold', color='darkred')
        
        # Y-axis (green-ish)
        ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', alpha=axis_alpha, linewidth=axis_width)
        ax.plot([0, 0], [-axis_length, 0], [0, 0], 'g--', alpha=axis_alpha/2, linewidth=axis_width)
        ax.text(0, axis_length, 0, 'Y', fontsize=14, fontweight='bold', color='darkgreen')
        
        # Z-axis (blue-ish)
        ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', alpha=axis_alpha, linewidth=axis_width)
        ax.plot([0, 0], [0, 0], [-max_range/5, 0], 'b--', alpha=axis_alpha/2, linewidth=axis_width)
        ax.text(0, 0, axis_length, 'Z', fontsize=14, fontweight='bold', color='darkblue')
    
    # Plot each vector as an arrow
    for i, (vec, color, label) in enumerate(zip(vectors, colors, labels)):
        # Main vector arrow
        ax.quiver(origin[0], origin[1], origin[2], 
                  vec[0], vec[1], vec[2], 
                  color=color, arrow_length_ratio=0.15, 
                  linewidth=3, alpha=0.9, label=label)
        
        # Add text labels at the end of each vector
        ax.text(vec[0], vec[1], vec[2], f'  V{i+1}', 
                color=color, fontsize=11, fontweight='bold')
        
        # Add projection lines if requested
        if show_projections:
            # Project to XY plane
            ax.plot([vec[0], vec[0]], [vec[1], vec[1]], [0, vec[2]], 
                    color=color, linestyle=':', alpha=0.3, linewidth=1)
            # Project to XZ plane  
            ax.plot([vec[0], vec[0]], [0, vec[1]], [vec[2], vec[2]], 
                    color=color, linestyle=':', alpha=0.3, linewidth=1)
            # Project to YZ plane
            ax.plot([0, vec[0]], [vec[1], vec[1]], [vec[2], vec[2]], 
                    color=color, linestyle=':', alpha=0.3, linewidth=1)
    
    # Plot origin point
    ax.scatter([0], [0], [0], color='black', s=200, alpha=1, marker='o', 
               edgecolors='white', linewidth=2, label='Origin (0,0,0)', zorder=10)
    
    # Add text label for origin
    ax.text(0.1, 0.1, 0.1, 'Origin\n(0,0,0)', fontsize=10, fontweight='bold', 
            color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add coordinate plane at z=0 if requested
    if show_plane:
        xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 10), 
                             np.linspace(-max_range, max_range, 10))
        z_plane = np.zeros_like(xx)
        
        # XY plane at z=0 - gray transparent surface
        ax.plot_surface(xx, yy, z_plane, alpha=0.1, color='gray')
        
        # Add gridlines on the z=0 plane
        if show_grid:
            for i in np.linspace(-max_range, max_range, 11):
                ax.plot([i, i], [-max_range, max_range], [0, 0], 'gray', alpha=0.2, linewidth=0.5)
                ax.plot([-max_range, max_range], [i, i], [0, 0], 'gray', alpha=0.2, linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Z', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range/5, max_range])
    
    # Add grid
    ax.grid(show_grid, alpha=0.2, linestyle='--')
    
    # Add tick marks at integer positions
    tick_range = int(max_range)
    ax.set_xticks(range(-tick_range, tick_range + 1))
    ax.set_yticks(range(-tick_range, tick_range + 1))
    ax.set_zticks(range(0, tick_range + 1))
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Set viewing angle
    ax.view_init(elev=view_elev, azim=view_azim)
    
    # Style the background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    
    # Style the axes
    ax.xaxis.line.set_color('gray')
    ax.yaxis.line.set_color('gray')
    ax.zaxis.line.set_color('gray')
    ax.tick_params(colors='gray')
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    else:
        plt.show()
        
    
def print_vector_info(vectors):
    """
    Print information about the vectors.
    
    Parameters:
    -----------
    vectors : torch.Tensor, np.ndarray, or list
        Matrix where each row is a 3D vector
    """
    # Convert to numpy if needed
    if torch.is_tensor(vectors):
        vectors = vectors.numpy()
    elif isinstance(vectors, list):
        vectors = np.array(vectors)
    
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    
    print("Vector Information:")
    print("-" * 40)
    for i, vec in enumerate(vectors):
        magnitude = np.linalg.norm(vec)
        print(f"Vector {i+1}: [{vec[0]:6.2f}, {vec[1]:6.2f}, {vec[2]:6.2f}]  |  Magnitude: {magnitude:.3f}")


# Example usage function
def example_usage():
    """
    Example of how to use the vector visualization functions.
    """
    # Using torch tensor
    tensor = torch.tensor([[ 1., 0., 0.], 
                           [-2., 1., 0.], 
                           [-1., 1., 1.]])
    
    # Basic plot
    plot_3d_vectors(tensor)
    
    # Customized plot
    plot_3d_vectors(
        tensor,
        colors=['crimson', 'forestgreen', 'royalblue'],
        title='My Custom Vector Plot',
        view_elev=30,
        view_azim=45,
        show_projections=False
    )
    
    # Print vector information
    print_vector_info(tensor)


if __name__ == "__main__":
    # Run example if this file is executed directly
    example_usage()