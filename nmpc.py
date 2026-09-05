import json
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import casadi as ca
import numpy as np

from constants import (
    C_m0,
    C_r0,
    C_r1,
    C_r2,
    Nf,
    T_max,
    car_length,
    car_width,
    delta_max,
    m,
    nu,
    nx,
    wheelbase,
)
from utils import FloatArray, rk4

# Cost weights
q_lon: float = 10.0
q_lat: float = 20.0
q_phi: float = 50.0
q_v: float = 20.0
r_T: float = 1e-3
r_delta: float = 2.0
q_lon_f: float = 1000.0
q_lat_f: float = 1000.0
q_phi_f: float = 500.0
q_v_f: float = 1000.0


@dataclass
class ControllerStats:
    runtime: float
    cost: float
    num_iters: int = 0


def get_continuous_dynamics() -> ca.Function:
    X = ca.SX.sym("X")
    Y = ca.SX.sym("Y")
    phi = ca.SX.sym("phi")
    v = ca.SX.sym("v")
    T = ca.SX.sym("T")
    delta = ca.SX.sym("delta")
    x = ca.vertcat(X, Y, phi, v)
    u = ca.vertcat(T, delta)
    beta = 0.5 * delta
    v_x = v * ca.cos(beta)
    l_R = 0.5 * wheelbase
    return ca.Function(
        "continuous_dynamics",
        [x, u],
        [
            ca.vertcat(
                v * ca.cos(phi + beta),
                v * ca.sin(phi + beta),
                v * ca.sin(beta) / l_R,
                (C_m0 * T - (C_r0 + C_r1 * v_x + C_r2 * v_x**2) * ca.tanh(10 * v_x))
                / m,
            )
        ],
    )


def get_discrete_dynamics() -> ca.Function:
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    return ca.Function(
        "discrete_dynamics", [x, u], [ca.cse(rk4(get_continuous_dynamics(), x, u))]
    )


class NMPCController:
    def __init__(
        self, solver: Literal["fatrop", "ipopt"] = "fatrop", jit: bool = False
    ):
        self.solver = solver
        self.discrete_dynamics = get_discrete_dynamics()

        opti = ca.Opti()
        # Variables must be declared interleaved (x[i], u[i] per stage) for fatrop
        # to correctly detect the block-sparse OCP structure.
        x = []
        u = []
        for i in range(Nf):
            x.append(opti.variable(nx))
            u.append(opti.variable(nu))
        x.append(opti.variable(nx))

        x0 = opti.parameter(nx)
        X_ref = opti.parameter(Nf + 1)
        Y_ref = opti.parameter(Nf + 1)
        phi_ref = opti.parameter(Nf + 1)
        v_ref = opti.parameter(Nf + 1)

        # Build cost function
        cost = 0.0
        all_left_e_lat = []
        all_right_e_lat = []
        for i in range(Nf):
            T_k, delta_k = ca.vertsplit(u[i], 1)
            T_ref_k = (
                ca.tanh(10 * v_ref[i])
                * (C_r0 + C_r1 * v_ref[i] + C_r2 * v_ref[i] ** 2)
                / C_m0
            )
            cost += r_T * (T_k - T_ref_k) ** 2 + r_delta * delta_k**2
            if i > 0:
                cp, sp = ca.cos(phi_ref[i]), ca.sin(phi_ref[i])
                X_k, Y_k, phi_k, v_k = ca.vertsplit(x[i], 1)
                e_lon = cp * (X_k - X_ref[i]) + sp * (Y_k - Y_ref[i])
                e_lat = -sp * (X_k - X_ref[i]) + cp * (Y_k - Y_ref[i])
                all_left_e_lat.append(
                    e_lat
                    + 0.5 * car_width * ca.cos(phi_k - phi_ref[i])
                    + 0.5 * car_length * ca.sin(phi_k - phi_ref[i])
                )
                all_right_e_lat.append(
                    e_lat
                    - 0.5 * car_width * ca.cos(phi_k - phi_ref[i])
                    + 0.5 * car_length * ca.sin(phi_k - phi_ref[i])
                )
                cost += (
                    q_lon * e_lon**2
                    + q_lat * e_lat**2
                    + q_phi * (phi_k - phi_ref[i]) ** 2
                    + q_v * (v_k - v_ref[i]) ** 2
                )

        cp, sp = ca.cos(phi_ref[Nf]), ca.sin(phi_ref[Nf])
        X_f, Y_f, phi_f, v_f = ca.vertsplit(x[Nf], 1)
        e_lon = cp * (X_f - X_ref[Nf]) + sp * (Y_f - Y_ref[Nf])
        e_lat = -sp * (X_f - X_ref[Nf]) + cp * (Y_f - Y_ref[Nf])
        all_left_e_lat.append(
            e_lat
            + 0.5 * car_width * ca.cos(phi_f - phi_ref[Nf])
            + 0.5 * car_length * ca.sin(phi_f - phi_ref[Nf])
        )
        all_right_e_lat.append(
            e_lat
            - 0.5 * car_width * ca.cos(phi_f - phi_ref[Nf])
            + 0.5 * car_length * ca.sin(phi_f - phi_ref[Nf])
        )
        cost += (
            q_lon_f * e_lon**2
            + q_lat_f * e_lat**2
            + q_phi_f * (phi_f - phi_ref[Nf]) ** 2
            + q_v_f * (v_f - v_ref[Nf]) ** 2
        )
        cost = ca.cse(cost)
        opti.minimize(cost)

        # Constraints (order matters for fatrop)
        for i in range(Nf):
            opti.subject_to(x[i + 1] == self.discrete_dynamics(x[i], u[i]))
            if i == 0:
                opti.subject_to(x[i] == x0)
            opti.subject_to((-T_max <= u[i][0]) <= T_max)
            opti.subject_to((-delta_max <= u[i][1]) <= delta_max)
            if i > 0:
                opti.subject_to((-1.25 <= all_left_e_lat[i - 1]) <= 1.25)
                opti.subject_to((-1.25 <= all_right_e_lat[i - 1]) <= 1.25)
                opti.subject_to((0.0 <= x[i][3]) <= 5.0)
        opti.subject_to((-1.25 <= all_left_e_lat[Nf - 1]) <= 1.25)
        opti.subject_to((-1.25 <= all_right_e_lat[Nf - 1]) <= 1.25)
        opti.subject_to((0.0 <= x[Nf][3]) <= 5.0)

        # Solver options
        if solver == "ipopt":
            options = {
                "print_time": 0,
                "expand": True,
                "ipopt": {"sb": "yes", "print_level": 0, "max_resto_iter": 0},
            }
        elif solver == "fatrop":
            options = {
                "print_time": 0,
                "debug": True,
                "expand": True,
                "structure_detection": "auto",
                "fatrop": {"print_level": 0},
            }
        else:
            options = {
                "expand": True,
                "qpsol": "osqp",
                "print_time": 0,
                "print_header": False,
                "print_iteration": False,
                "print_status": False,
                "max_iter": 1,
                "qpsol_options": {"osqp": {"verbose": False}},
            }
        options.update(
            {
                "jit": jit,
                "jit_options": {"flags": ["-O3 -march=native"], "verbose": False},
            }
        )
        opti.solver(solver, {**options, "jit": False})
        callbacks = opti.to_function(
            "callbacks", [opti.x, opti.p], [opti.x]
        ).find_function("solver")
        options["cache"] = {
            name: callbacks.get_function(name).transform([["simplify", "cse"]])
            for name in callbacks.get_function()
            if name.startswith("nlp_")
        }
        opti.solver(solver, options)

        self.opti = opti
        self.x0 = x0
        self.X_ref = X_ref
        self.Y_ref = Y_ref
        self.phi_ref = phi_ref
        self.v_ref = v_ref
        self.x = x
        self.u = u
        self.cost = cost

        if jit:
            print("Generating JIT code... ", end="", flush=True)
            start = perf_counter()
            self.control(
                0.0,
                0.0,
                0.0,
                0.0,
                np.zeros(Nf + 1),
                np.zeros(Nf + 1),
                np.zeros(Nf + 1),
                np.zeros(Nf + 1),
            )
            print(f"done in {perf_counter() - start:.2f} s")

    def codegen(self):
        states = ca.horzcat(*self.x)
        controls = ca.horzcat(*self.u)
        solver_function = self.opti.to_function(
            f"nmpc_solver_{self.solver}",
            [
                self.x0,
                self.X_ref,
                self.Y_ref,
                self.phi_ref,
                self.v_ref,
                states,
                controls,
            ],
            [states, controls, self.cost],
            ["x0", "X_ref", "Y_ref", "phi_ref", "v_ref", "x_guess", "u_guess"],
            ["x_opt", "u_opt", "cost"],
        )
        solver_function.generate(
            f"nmpc_solver_{self.solver}",
            {
                "with_mem": True,
                "force_canonical": True,
                "with_header": True,
                "verbose": True,
                "indent": 4,
                "main": True,
            },
        )

    def control(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ) -> tuple[FloatArray, FloatArray, ControllerStats]:
        # Set parameters
        self.opti.set_value(self.x0, np.array([X, Y, phi, v]))
        self.opti.set_value(self.X_ref, X_ref)
        self.opti.set_value(self.Y_ref, Y_ref)
        self.opti.set_value(self.phi_ref, phi_ref)
        self.opti.set_value(self.v_ref, v_ref)

        # Warm-start from reference
        T_ref = (C_r0 + C_r1 * v_ref + C_r2 * v_ref * v_ref) / C_m0
        for i in range(Nf):
            self.opti.set_initial(
                self.x[i], np.array([X_ref[i], Y_ref[i], phi_ref[i], v_ref[i]])
            )
            self.opti.set_initial(self.u[i], np.array([T_ref[i], 0.0]))
        self.opti.set_initial(
            self.x[Nf], np.array([X_ref[Nf], Y_ref[Nf], phi_ref[Nf], v_ref[Nf]])
        )

        # Solve
        start = perf_counter()
        try:
            sol = self.opti.solve_limited()
        except RuntimeError as err:
            with open("nmpc_inputs.json", "w") as f:
                json.dump(
                    {
                        "X": X,
                        "Y": Y,
                        "phi": phi,
                        "v": v,
                        "X_ref": X_ref.tolist(),
                        "Y_ref": Y_ref.tolist(),
                        "phi_ref": phi_ref.tolist(),
                        "v_ref": v_ref.tolist(),
                    },
                    f,
                )
            raise err
        runtime = perf_counter() - start

        cost = sol.value(self.cost)
        x_pred = np.array([sol.value(self.x[i]) for i in range(Nf + 1)])
        u_pred = np.array([sol.value(self.u[i]) for i in range(Nf)])

        stats = self.opti.stats()
        if (
            not stats["success"]
            and stats["return_status"] != "Maximum_Iterations_Exceeded"
        ):
            print(stats)
            raise RuntimeError(stats["return_status"])

        return (
            x_pred,
            u_pred,
            ControllerStats(runtime=runtime, cost=cost, num_iters=stats["iter_count"]),
        )
