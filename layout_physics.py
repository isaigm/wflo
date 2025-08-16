import numpy as np

D = 40.0
R = D / 2.0
Z = 60.0
Zo = 0.3
Ct = 0.88

# \[ a = \frac{1}{2} \left(1 - \sqrt{1 - C_T}\right) \] 
a = 0.5 * (1 - np.sqrt(1 - Ct))
#  \[\alpha = \frac{0.5}{\ln(Z/Z_0)}\] 
alpha = 0.5 / np.log(Z / Zo)


# \[\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}\]
def rotate_coordinates(layout, angle_deg, center):
    layout_centered = layout - center
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a,  cos_a]])
    rotated_centered_layout = layout_centered @ rotation_matrix.T
    return rotated_centered_layout + center


#\[u = u_0 \left( 1 - \frac{2a}{(1 + \alpha \frac{x}{r_d})^2} \right)\] 
def get_wake_velocity(u_freestream, x_dist, r_d_wake):
    denominator = (1 + alpha * (x_dist / r_d_wake))**2
    velocity_deficit = (2 * a) / denominator
    return u_freestream * (1 - velocity_deficit)


#\[P(u) = \begin{cases} 0 & u < 3 \text{ or } u > 25 \\ 0.3u^3 & 3 \le u < 12 \\ 518.4 & 12 \le u \le 25  \end{cases}\]
def get_power(u):
    u = np.atleast_1d(u)
    power = np.zeros_like(u, dtype=float)
    cut_in_speed =  3.0
    rated_speed = 12.0
    cut_out_speed = 25.0
    rated_power = 518.4
    mask1 = (u >= cut_in_speed) & (u < rated_speed)
    power[mask1] = 0.3 * u[mask1]**3
    mask2 = (u >= rated_speed) & (u <= cut_out_speed)
    power[mask2] = rated_power
    return power.item() if power.size == 1 else power



def evaluate_layout(layout, wind_direction_deg, u_freestream, farm_dims):
    farm_center = np.array([farm_dims[0] / 2, farm_dims[1] / 2])
    rotated_layout = rotate_coordinates(layout, -wind_direction_deg, center=farm_center)
    sorted_indices = np.argsort(rotated_layout[:, 1])[::-1]
    sorted_rotated_layout = rotated_layout[sorted_indices]
    n_turbines = len(sorted_rotated_layout)
    turbine_velocities = np.full(n_turbines, u_freestream)
    # \[ r_d = r_0 \sqrt{\frac{1-a}{1-2a}}\]
    r_d = R * np.sqrt((1 - a) / (1 - 2 * a))
    for i in range(n_turbines):
        pos_i = sorted_rotated_layout[i]    
        sum_of_squared_deficits = 0.0
        for j in range(i):
            pos_j = sorted_rotated_layout[j]
            dist_downwind = pos_j[1] - pos_i[1]
            if dist_downwind > 0:
                dist_crosswind = np.abs(pos_j[0] - pos_i[0])
                wake_radius_at_i = r_d + alpha * dist_downwind
                if dist_crosswind < wake_radius_at_i:
                    u_wake_base = get_wake_velocity(u_freestream, dist_downwind, r_d)
                    base_deficit = (u_freestream - u_wake_base) / u_freestream
                    sum_of_squared_deficits += base_deficit**2
        if sum_of_squared_deficits > 0:
            total_deficit = np.sqrt(sum_of_squared_deficits)
            turbine_velocities[i] = u_freestream * (1 - total_deficit)
    return np.sum(get_power(turbine_velocities))


def create_grid(rows, cols, width, height):
        cell_width = width / cols
        cell_height = height / rows
        x_coords = np.linspace(cell_width / 2, width - cell_width / 2, cols)
        y_coords = np.linspace(cell_height / 2, height - cell_height / 2, rows)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        return np.vstack([grid_x.ravel(), grid_y.ravel()]).T