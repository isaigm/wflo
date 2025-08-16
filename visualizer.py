import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from layout_physics import R, a, alpha, get_wake_velocity


def plot_layout(layout, scenario, title="Wind Farm Layout"):
    """
    Draws a plot of the turbine layout, their wakes, and the grid of possible locations.
    This version takes an OptimizationScenario object for clarity and robustness.
    """
    farm_dims = scenario.farm_dims
    full_grid = scenario.grid_to_use
    wind_direction_deg = scenario.wind_direction
    u_freestream = scenario.u_freestream
    
    farm_width, farm_height = farm_dims
    
    min_x, max_x = np.min(full_grid[:, 0]), np.max(full_grid[:, 0])
    min_y, max_y = np.min(full_grid[:, 1]), np.max(full_grid[:, 1])
    
    margin = 200 
    plot_xlim = (min_x - margin, max_x + margin)
    plot_ylim = (min_y - margin, max_y + margin)
    
    plot_width = plot_xlim[1] - plot_xlim[0]
    plot_height = plot_ylim[1] - plot_ylim[0]
    fig_height = max(5, 10 * (plot_height / plot_width))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    ax.scatter(full_grid[:,0], full_grid[:,1], c='gray', s=5, alpha=0.5, zorder=1)

    r_d = R * np.sqrt((1 - a) / (1 - 2 * a))
    colormap = cm.get_cmap('inferno')
    max_deficit = (u_freestream - get_wake_velocity(u_freestream, x_dist=0.1, r_d_wake=r_d)) / u_freestream
    norm = mcolors.Normalize(vmin=0, vmax=max_deficit)
    wake_length = max(plot_width, plot_height)
    num_segments = 30
    segment_length = wake_length / num_segments
    
    for i in range(len(layout)):
        for seg in range(num_segments):
            dist_start, dist_end = seg * segment_length, (seg + 1) * segment_length
            R_start, R_end = r_d + alpha * dist_start, r_d + alpha * dist_end
            base_vertices = np.array([[-R_start, -dist_start], [+R_start, -dist_start], [+R_end, -dist_end], [-R_end, -dist_end]])
            plot_angle_rad = np.radians(-wind_direction_deg)
            rotation_matrix = np.array([[np.cos(plot_angle_rad), -np.sin(plot_angle_rad)], [np.sin(plot_angle_rad), np.cos(plot_angle_rad)]])
            rotated_vertices = base_vertices @ rotation_matrix.T + layout[i]
            avg_dist = (dist_start + dist_end) / 2
            current_deficit = (u_freestream - get_wake_velocity(u_freestream, avg_dist, r_d)) / u_freestream
            segment_color = colormap(norm(current_deficit))
            wake_segment = patches.Polygon(rotated_vertices, closed=True, edgecolor='none', facecolor=segment_color, zorder=2)
            ax.add_patch(wake_segment)
            
    # --- Plotting the Turbines ---
    ax.scatter(layout[:, 0], layout[:, 1], c='cyan', marker='o', s=50, edgecolor='black', zorder=5)
    
    # --- Plot Formatting ---
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Wake Intensity (Velocity Deficit)', rotation=270, labelpad=15)
    
    ax.set_facecolor('black')
    fig.set_facecolor('black')
    ax.set_title(title, color='white')
    ax.set_xlabel("X Coordinate (m)", color='white')
    ax.set_ylabel("Y Coordinate (m)", color='white')
    
    ax.set_aspect('equal')
    ax.set_xlim(plot_xlim)
    ax.set_ylim(plot_ylim)
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.yaxis, 'ticklabels'), color='white')
    
    plt.show()