# Minimal Tracking NMPC

Closed-loop center-line tracking for a Formula Student car using a Nonlinear MPC solved with CasADi (ipopt or fatrop).

## Usage

Run from this directory so that the `data/` symlink resolves correctly.

```bash
# Run simulation (saves to closed_loop_data.npz)
uv run python main.py run --solver ipopt --track fsds_competition_1 --vref 5.0

# Run and immediately visualize
uv run python main.py run --solver ipopt --viz

# Visualize a previously saved result
uv run python main.py viz --data-file closed_loop_data.npz

# Plot the track and motion planner reference
uv run python main.py track --track fsds_competition_1

# Generate a standalone C solver
uv run python main.py codegen --solver ipopt
```

## System model

Kinematic bicycle model with state $x = (X, Y, \phi, v)$ and control $u = (T, \delta)$:

$$
\dot{X} = v \cos(\phi + \beta), \quad
\dot{Y} = v \sin(\phi + \beta), \quad
\dot{\phi} = \frac{v \sin\beta}{L/2}, \quad
\dot{v} = \frac{F_x}{m}
$$

where $\beta = \delta/2$ is the kinematic slip angle, $L$ the wheelbase, $m$ the vehicle mass, and

$$
F_x = C_{m0}\, T - \bigl(C_{r0} + C_{r1} v_x + C_{r2} v_x^2\bigr)\tanh(10\,v_x)
$$

the longitudinal force, with $v_x = v \cos\beta$.

The continuous dynamics are discretised with RK4 at $\Delta t = 0.05\,\text{s}$.

## NMPC formulation

The OCP solved at each time step, given a reference trajectory $\{x_k^{\text{ref}}\}_{k=0}^{N_f}$ and current state $x_0$, is:

$$
\min_{x_{0:N_f},\, u_{0:N_f-1}} \quad F(x_{N_f}, x_{N_f}^{\text{ref}}) + \sum_{k=0}^{N_f-1} \ell(x_k, u_k, x_k^{\text{ref}})
$$

$$
\text{s.t.} \quad x_{k+1} = f(x_k, u_k), \quad k = 0,\ldots,N_f-1
$$

$$
x_0 = \hat{x}, \quad -T_{\max} \le T_k \le T_{\max}, \quad -\delta_{\max} \le \delta_k \le \delta_{\max}
$$

### Position error decomposition

To enable independent lateral/longitudinal tuning, errors are expressed in the frame of the reference heading $\phi^{\text{ref}}$:

$$
e_{\text{lon}} = \cos\phi^{\text{ref}}(X - X^{\text{ref}}) + \sin\phi^{\text{ref}}(Y - Y^{\text{ref}})
$$

$$
e_{\text{lat}} = -\sin\phi^{\text{ref}}(X - X^{\text{ref}}) + \cos\phi^{\text{ref}}(Y - Y^{\text{ref}})
$$

### Track constraints

The lateral footprint of the car (half-width $w/2$, half-length $\ell/2$) must stay within $\pm 1.25\,\text{m}$ of the centerline:

$$
-1.25 \le e_{\text{lat}} \pm \tfrac{w}{2}\cos(\phi - \phi^{\text{ref}}) + \tfrac{\ell}{2}\sin(\phi - \phi^{\text{ref}}) \le 1.25
$$

### Stage cost

The reference throttle for steady-state at $v^{\text{ref}}$ is

$$
T^{\text{ref}} = \frac{C_{r0} + C_{r1} v^{\text{ref}} + C_{r2} (v^{\text{ref}})^2}{C_{m0}}
$$

and the stage cost is

$$
\ell(x, u, x^{\text{ref}}) = q_{\text{lon}}\, e_{\text{lon}}^2 + q_{\text{lat}}\, e_{\text{lat}}^2 + q_\phi\,(\phi - \phi^{\text{ref}})^2 + q_v\,(v - v^{\text{ref}})^2 + r_T\,(T - T^{\text{ref}})^2 + r_\delta\,\delta^2
$$

### Terminal cost

$$
F(x, x^{\text{ref}}) = q_{\text{lon},f}\, e_{\text{lon}}^2 + q_{\text{lat},f}\, e_{\text{lat}}^2 + q_{\phi,f}\,(\phi - \phi^{\text{ref}})^2 + q_{v,f}\,(v - v^{\text{ref}})^2
$$

### Cost weights

| Weight | Value |
|--------|-------|
| $q_{\text{lon}}$ | 10 |
| $q_{\text{lat}}$ | 20 |
| $q_\phi$ | 50 |
| $q_v$ | 20 |
| $r_T$ | 0.001 |
| $r_\delta$ | 2 |
| $q_{\text{lon},f}$ | 1000 |
| $q_{\text{lat},f}$ | 1000 |
| $q_{\phi,f}$ | 500 |
| $q_{v,f}$ | 1000 |

## Motion planner

A minimum-curvature cubic spline is fitted offline to the track centerline waypoints. At each time step the car's arc-length position is projected onto the spline, and a constant-velocity reference trajectory is sampled uniformly in time over the MPC horizon.

## Dependencies

- `numpy`, `scipy`
- `qpsolvers[osqp]` (spline fitting QP)
- `casadi` (NLP formulation and solvers)
- `matplotlib`, `tqdm`
