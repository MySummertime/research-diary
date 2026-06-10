# Algorithm Analysis

## 1. Dantzig-Wolfe Decomposition: The Mathematical Blueprint

Faced with the massive symmetry problem caused by homogeneous vehicles, directly solving the original model is infeasible. Dantzig-Wolfe decomposition is a mathematical reformulation idea (Blueprint) that splits the originally large and messy model into two parts:

- **Master Problem**: Only responsible for "big-picture selection" — from a known pool of feasible truck routes, it selects the optimal combination to satisfy total customer demand while staying within budget.
- **Pricing Subproblem**: Responsible for "microscopic exploration" — considering extremely complex physical networks, time-varying accident risk thresholds, and loading/unloading time windows, it independently searches for lower-cost legal routes for a single truck.

**Core significance**: This decomposition blueprint inherently eliminates the mathematical symmetry of vehicle scheduling.

## 2. Column Generation (CG): The Core Computation Engine for LP Relaxation

Based on the Dantzig-Wolfe decomposition blueprint, we cannot enumerate all millions of legal truck routes (i.e., "columns") and write them into the master problem from the start — memory would overflow immediately. Therefore, the Column Generation algorithm is introduced as the actual iterative computation Engine:

1. Initially, the master problem (Restricted Master Problem, RMP) contains only a very small number of initial routes. It solves a temporary linear programming (LP) solution and produces dual variables (i.e., shadow prices, such as the huge penalty for unsatisfied demand at a certain customer).
2. Dual variables are sent down to the subproblem as "bounty rewards." The subproblem (using a bidirectional label-setting algorithm, ESPPRC) searches the enormous space-time network for new routes with negative reduced cost (i.e., "columns worth adding to the master problem").
3. When found, they are reported to the master problem and added to the column pool. The master problem re-solves. This iterates until the subproblem can no longer find routes with negative reduced cost — at which point the algorithm has mathematically rigorous proof of finding the global optimum of this LP relaxation among millions of potential routes.

## 3. Branch-and-Price (B&P): The Grand Commander for Finding Integer Solutions

Column Generation (CG) is very powerful, but it has a fatal weakness: the route selection variables (λ) it produces may be fractional (e.g., 0.5 trucks take Route A, 0.5 trucks take Route B), which is physically inexecutable — we need strict integer solutions.

To obtain integer solutions, the traditional Branch-and-Bound Tree must be activated. However, because we previously used Column Generation, after each branching decision (e.g., forcing a truck to pass through / not pass through a certain node), the existing column pool may no longer be complete. Therefore, at every node of the branching tree, the Column Generation engine (CG) must be restarted to find new routes.

This complete framework — "embedding Column Generation at every node of the Branch-and-Bound tree" — is known in the academic literature as **Branch-and-Price (B&P)**.
