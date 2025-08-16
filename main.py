import numpy as np
from mealpy import IntegerVar, GA
from layout_physics import evaluate_layout, rotate_coordinates, get_power, create_grid
import visualizer  
import pandas as pd

class OptimizationScenario:
    def __init__(self, farm_dims, n_turbines, wind_direction, u_freestream, grid_to_use):
        self.farm_dims = farm_dims
        self.n_turbines = n_turbines
        self.wind_direction = wind_direction
        self.u_freestream = u_freestream
        self.grid_to_use = grid_to_use 

    def objective_func(self, solution):
        indices = solution.astype(int)
        unique_indices = np.unique(indices)
        if len(unique_indices) < self.n_turbines:
            return 1e9
        layout = self.grid_to_use[unique_indices]
        power = evaluate_layout(layout, wind_direction_deg=self.wind_direction, u_freestream=self.u_freestream, farm_dims=self.farm_dims)
        return -power

if __name__ == "__main__":
    
    SCENARIO = 'PROPOSED' 
    
    if SCENARIO == 'PROPOSED':
        farm_dims = (2000.0, 2000.0)
        base_grid = create_grid(rows=10, cols=10, width=farm_dims[0], height=farm_dims[1])
        grid_to_use = rotate_coordinates(base_grid, 45, center=np.array(farm_dims)/2)
        benchmark_power = 16.33
        benchmark_eff = 98.42
        benchmark_wake_loss = 262.20
        plot_title = "Best Layout in Rotated Grid"
        
    elif SCENARIO == 'REF_11':
        farm_dims = (2000.0, 2200.0)
        grid_to_use = create_grid(rows=11, cols=10, width=farm_dims[0], height=farm_dims[1])
        benchmark_power = 15.218
        benchmark_eff = 91.74 # Corregido el typo aquí
        benchmark_wake_loss = 1370.60
        plot_title = "Best Layout in 10x11 Grid (WFLO of [11])"
       
    scenario = OptimizationScenario(
        farm_dims=farm_dims,
        n_turbines=32,
        wind_direction=0.0,
        u_freestream=12.0,
        grid_to_use=grid_to_use
    )
    
    problem_dict = {
        "obj_func": scenario.objective_func,
        "bounds": IntegerVar(lb=[0] * scenario.n_turbines, ub=[len(scenario.grid_to_use) - 1] * scenario.n_turbines),
        "minmax": "min",
        "verbose": True,
    }

    optimizer = GA.BaseGA(epoch=2000, pop_size=350, pm=0.09)   
    agent = optimizer.solve(problem_dict)

    # --- Process and Validate Results ---
    best_indices = agent.solution.astype(int)
    final_indices = np.unique(best_indices)
    if len(final_indices) < scenario.n_turbines:
        available_indices = np.setdiff1d(np.arange(len(scenario.grid_to_use)), final_indices)
        needed = scenario.n_turbines - len(final_indices)
        final_indices = np.concatenate([final_indices, np.random.choice(available_indices, needed, replace=False)])

    best_layout = scenario.grid_to_use[final_indices]
    best_power = evaluate_layout(best_layout, scenario.wind_direction, scenario.u_freestream, scenario.farm_dims)
    
    # --- Corregido: Extraer el escalar con [0] ---
    power_per_turbine_ideal = get_power(np.array([scenario.u_freestream]))
    ideal_power = power_per_turbine_ideal * scenario.n_turbines
   
    wake_loss_kw = ideal_power - best_power
    efficiency = (best_power / ideal_power) * 100
    
    df = pd.DataFrame({
        "Metric": ["Total Power (MW)", "Efficiency (%)", "Wake Loss (kW)"],
        "Our Result": [best_power / 1000, efficiency, wake_loss_kw],
        "Benchmark": [benchmark_power, benchmark_eff, benchmark_wake_loss]
    })
    print("\n--- VALIDATION RESULTS ---")
    print(df.to_string(index=False))
    
    visualizer.plot_layout(
        layout=best_layout,
        scenario=scenario,
        title=f"{plot_title} ({scenario.n_turbines} Turbines)"
    )