# From Curiosity to Code: A Deep Dive into Wind Farm Optimization

<img width="1000" height="810" alt="layout" src="https://github.com/user-attachments/assets/cee2818d-027d-439b-b355-7e356ab78763" />

This project is the result of a personal deep dive that began with a spark of curiosity and evolved into a full-blown engineering forensic analysis. It's the story of replicating, validating, and ultimately reverse-engineering the methodology behind the research paper "Optimal Placement of Wind Turbines in Wind Farm Layout Using Particle Swarm Optimization".

The core of the project is a simulation and optimization engine built from scratch in Python. It tackles the complex, high-dimensional problem of wind farm layout optimization, where minimizing the aerodynamic "wake effect" is critical for maximizing total energy yield.

## The Journey: From Paper to Validated Model

This project didn't follow a straight path. It was an iterative journey of building, testing, and critical analysis.

1.  **The Spark of Curiosity:** The project started with a fascination for the engineering challenge of wind farm optimization. I set myself a challenge: could I build a working model based on the research?

2.  **Building the Simulation Engine:** Recognizing this was a deep dive into a new domain, I leveraged generative AI as a coding co-pilot. This allowed me to direct the development of a physics-based simulator in Python, implementing the complex Jensen wake model to handle aerodynamic interactions between turbines.

3.  **Integrating the AI Optimizer:** The physics engine was then coupled with a Genetic Algorithm from the `mealpy` library. This enabled an AI agent to search for the most efficient turbine layouts by navigating a complex, high-dimensional search space.

4.  **The Quest for Ground Truth:** The biggest challenge wasn't building, but **validating.** The initial results were promising but didn't align with the paper's top-performing benchmarks. This led to a forensic investigation. I tracked down the 90-page Master's thesis referenced in the paper to find the hidden details of their methodology.

5.  **The Breakthrough:** The thesis revealed the crucial, unstated assumption: their best results came from a **combinatorial optimization on a pre-defined, 45-degree rotated grid**, not a free-form placement.

## The Final, Validated Result

By re-engineering the optimizer to match this exact methodology, the results finally clicked into place. When testing the paper's high-performance "Proposed Strategy", my implementation achieved **96.71% efficiency**, a near-perfect replication of their 98.42% benchmark. This confirmed the simulation engine's accuracy and proved that the geometric setup was the key.

| Metric                  | My Result        | Paper's Proposed Strategy |
| :---------------------- | :--------------- | :------------------------ |
| **Total Power (MW)**    | **16.047**       | 16.33                     |
| **Efficiency (%)**      | **96.71**        | 98.42                     |
| **Wake loss (kW)**      | **541.03**       | 262.20                    |

## Getting Started

### Prerequisites

This project requires Python 3.8+ and the following libraries:
-   NumPy
-   Matplotlib
-   mealpy
-   pandas

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/isaigm/wflo
    cd wflo
    ```
2.  Install the required packages:
    ```bash
    pip install numpy matplotlib mealpy pandas
    ```

### Running the Simulation

The `main.py` script is configured to run different scenarios. To switch between them, open the file and change the `SCENARIO` variable:

```python
# Choose between 'PROPOSED' (rotated grid) or 'REF_11' (standard grid)
SCENARIO = 'PROPOSED'
```

## Future Work

This project serves as a robust foundation, but the journey of optimization is never truly over. Future iterations of this work will focus on pushing the boundaries of performance and physical accuracy:

-   **High-Performance Computing with C++/CUDA:** The current Python implementation is excellent for rapid prototyping and validation. The next logical step is to port the core simulation engine (`evaluate_layout`) to C++. This would provide a significant performance boost, allowing for much larger-scale optimizations (more turbines, more generations). Furthermore, the problem is highly parallelizable, making it an ideal candidate for acceleration with **NVIDIA CUDA**, which would enable near real-time analysis of massive layouts.

-   **Advanced Wake Modeling:** While the Jensen model is a powerful baseline, future versions could incorporate more advanced, high-fidelity models (such as the Ainslie eddy viscosity model or CFD-based approaches) to capture turbulence effects more accurately.

-   **Multi-Objective Optimization:** The current model optimizes for a single objective (maximizing power). A future version could use multi-objective algorithms (like NSGA-II, also available in `mealpy` or `pagmo`) to find the optimal trade-off between maximizing power, minimizing cost, and reducing structural fatigue on the turbines.
