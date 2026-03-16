import numpy as np
from tqdm import trange

from constants import dt
from motion_planning import MotionPlanner
from nmpc import NMPCController, get_discrete_dynamics
from track_data import load_center_line, load_cones


def closed_loop(
  controller: NMPCController,
  Tsim: float = 70.0,
  v_ref: float = 5.0,
  track_name: str = "fsds_competition_1",
  data_file: str = "closed_loop_data.npz",
):
  """
  Run a closed-loop simulation with the NMPC controller and save results to an npz file.

  Saved arrays (all in Cartesian coordinates):
  - x_ref:    reference states [X, Y, phi, v], shape (Nsim, Nf+1, 4)
  - x_pred:   predicted states [X, Y, phi, v], shape (Nsim, Nf+1, 4)
  - u_pred:   control predictions [T, delta],  shape (Nsim, Nf, 2)
  - runtimes: solver wall-clock times,          shape (Nsim,)
  - costs:    solver objective values,          shape (Nsim,)
  """
  Nsim = int(Tsim / dt) + 1
  x_current = np.array([0.0, 0.0, np.pi / 2, 0.0])
  s_guess = 0.0
  all_x_ref = []
  all_x_pred = []
  all_u_pred = []
  all_runtimes = []
  all_costs = []
  discrete_dynamics = get_discrete_dynamics()

  center_line, _ = load_center_line(f"data/tracks/{track_name}/center_line.csv")
  blue_cones, yellow_cones, big_orange_cones, _, _, _ = load_cones(f"data/tracks/{track_name}/cones.csv")
  motion_planner = MotionPlanner(center_line, v_ref=v_ref)

  progress_bar = trange(Nsim)
  for i in progress_bar:
    X, Y, phi, v = x_current

    s_guess, X_ref, Y_ref, phi_ref, v_ref_traj = motion_planner.plan(X, Y, phi, s_guess)
    all_x_ref.append(np.column_stack((X_ref, Y_ref, phi_ref, v_ref_traj)))

    try:
      x_pred, u_pred, stats = controller.control(X, Y, phi, v, X_ref, Y_ref, phi_ref, v_ref_traj)
    except RuntimeError as e:
      print(f"Controller failed at iteration {i}: {e}")
      break

    progress_bar.set_description(f"Runtime: {1000 * stats.runtime:.2f} ms, cost: {stats.cost:.2f}")
    all_runtimes.append(stats.runtime)
    all_costs.append(stats.cost)
    all_x_pred.append(x_pred)
    all_u_pred.append(u_pred)

    x_current = discrete_dynamics(x_current, u_pred[0]).full().ravel()

    if s_guess > motion_planner.lap_length:
      print(f"Completed a lap in {i} iterations ({i * dt:.1f} s)")
      break

  np.savez(
    data_file,
    x_ref=np.array(all_x_ref),
    x_pred=np.array(all_x_pred),
    u_pred=np.array(all_u_pred),
    runtimes=np.array(all_runtimes),
    costs=np.array(all_costs),
    center_line=np.column_stack((motion_planner.X_ref, motion_planner.Y_ref)),
    blue_cones=blue_cones,
    yellow_cones=yellow_cones,
    big_orange_cones=big_orange_cones,
  )
