import numpy as np
from blobmodel import Model, DefaultBlobFactory
import fppanalysis.two_dim_velocity_estimates as td


def make_2d_realization(vx, vy, xpoints, ypoints, T=1000, K=1000, Lx=10, Ly=10, taup=1e10, dt=0.01):
  bf = DefaultBlobFactory(A_dist="deg", W_dist="deg", vx_dist="deg", vy_dist="deg", vy_parameter=vy, vx_parameter=vx)
  bm = Model(Nx=10, Ny=1, Lx=Lx, Ly=Ly, dt=dt, T=T, num_blobs=K, blob_shape='gauss', periodic_y=True, t_drain=taup, blob_factory=bf)
  update_geometry(xpoints, ypoints, bm)
  return bm.make_realization(speed_up=True, error=10e-2)


# Change the geometry of the 2d model to lie similar to APD data.
def update_geometry(x_grid, y_grid, model):
  x_matrix, y_matrix, t_matrix = np.meshgrid(x_grid, y_grid, model._geometry.t)
  model._geometry.x_matrix = x_matrix
  model._geometry.y_matrix = y_matrix
  model._geometry.t_matrix = t_matrix
  model._geometry.Ny = len(y_grid)
  model._geometry.Nx = len(x_grid)
  model._geometry.x = x_grid
  model._geometry.y = y_grid


def test_rad_and_pol():
  v, w = 1, 1
  ds = make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
  v_est, w_est = td.estimate_velocities_for_pixel(1, 1, ds)
  error = np.max([abs(v_est-v), abs(w_est-w)])
  assert error < 0.1, "Numerical error too big"


def test_rad_and_neg_pol():
  v, w = 1, -1
  ds = make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
  v_est, w_est = td.estimate_velocities_for_pixel(1, 1, ds)
  error = np.max([abs(v_est-v), abs(w_est-w)])
  assert error < 0.1, "Numerical error too big"


def test_rad_and_2pol():
  v, w = 1, 2
  ds = make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
  v_est, w_est = td.estimate_velocities_for_pixel(1, 1, ds)
  error = np.max([abs(v_est - v), abs(w_est - w)])
  assert error < 0.1, "Numerical error too big"
