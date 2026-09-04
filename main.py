"""
Minimal tracking NMPC — entry point.

Commands:
  run     Run the closed-loop simulation and optionally visualize results.
  viz     Visualize results from a previously saved .npz file.
  codegen Generate a standalone C solver.
  track   Plot the track and motion planner reference trajectory.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(127)
np.set_printoptions(precision=3, suppress=True, linewidth=200)


def cmd_run(args):
  from nmpc import NMPCController
  from closed_loop import closed_loop
  from viz import visualize_trajectories_from_file

  closed_loop(
    controller=NMPCController(solver=args.solver, jit=args.jit),
    track_name=args.track,
    v_ref=args.vref,
    data_file=args.output,
  )
  if args.viz:
    visualize_trajectories_from_file(data_file=args.output, image_file=args.output.replace(".npz", ".png"))


def cmd_viz(args):
  from viz import visualize_trajectories_from_file

  visualize_trajectories_from_file(data_file=args.data_file, image_file=args.image_file)


def cmd_codegen(args):
  from nmpc import NMPCController

  NMPCController(solver=args.solver).codegen()


def cmd_track(args):
  from track_data import load_center_line, load_cones
  from motion_planning import MotionPlanner

  center_line, _ = load_center_line(f"data/tracks/{args.track}/center_line.csv")
  blue_cones, yellow_cones, big_orange_cones, small_orange_cones, _, _ = load_cones(f"data/tracks/{args.track}/cones.csv")
  motion_planner = MotionPlanner(center_line, v_ref=5.0)
  motion_planner.plot_motion_plan(center_line, blue_cones, yellow_cones, big_orange_cones, small_orange_cones, "Motion Planner")
  plt.show()


def main():
  parser = argparse.ArgumentParser(description="Minimal tracking NMPC")
  sub = parser.add_subparsers(dest="command", required=True)

  p_run = sub.add_parser("run", help="Run closed-loop simulation")
  p_run.add_argument("--solver", default="ipopt", choices=["ipopt", "fatrop", "sqpmethod"])
  p_run.add_argument("--jit", action="store_true")
  p_run.add_argument("--track", default="fsds_competition_1")
  p_run.add_argument("--vref", type=float, default=5.0)
  p_run.add_argument("--output", default="closed_loop_data.npz")
  p_run.add_argument("--viz", action="store_true", help="Visualize after simulation")

  p_viz = sub.add_parser("viz", help="Visualize results from file")
  p_viz.add_argument("--data-file", default="closed_loop_data.npz")
  p_viz.add_argument("--image-file", default="")

  p_cg = sub.add_parser("codegen", help="Generate standalone C solver")
  p_cg.add_argument("--solver", default="ipopt", choices=["ipopt", "fatrop"])

  p_track = sub.add_parser("track", help="Plot track and motion planner reference")
  p_track.add_argument("--track", default="fsds_competition_1")

  args = parser.parse_args()
  {"run": cmd_run, "viz": cmd_viz, "codegen": cmd_codegen, "track": cmd_track}[args.command](args)


if __name__ == "__main__":
  main()
