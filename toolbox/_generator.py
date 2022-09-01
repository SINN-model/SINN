#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def FPU(
    gamma=1, beta=1, U0=5, x0=1,sigma=0, dt=0.001, dt_save=0.2, t_burnin=10, t_total=100, batch=400
):
    '''
    Fermi-Pasta-Ulam potential (double-well)

    Parameters
    ----------
    gamma: float > 0
        coupling constant
    beta: float > 0
        inverse temperature
    U0: float > 0
         depth of the well
    x0: float > 0 
         location of the basin
    sigma: float > 0
        skewness of the double well 
    dt: float > 0
        time step size of the numerial solver 
    dt_save: float >= dt > 0
        time between saving snapshots
    t_burnin: float in (0, t_total)
        Length of trajectory to discard without saving
    t_total: float > 0
        Total time to simulate for each trajectory
    batch: int > 0
        Number of independent trajectories to generate

    Returns
    -------
    traj: tensor of size (batch, t_total / dt_save, 2)
        Sample trajectories, [:, :, 0] are momemtum, [:, :, 1] are position.
    '''

    def V_double_well(x,U0,x0,sigma):
        return 4*U0/x0**4*x**3-4*U0/x0**2*x+sigma/(2*x0)

    # t_total = 10*delta_t*sec_length
    # length=int(t_total/dt)+1
    # t = np.linspace(0,t_total,length) # define time axis
    total_steps = int(t_total // dt)
    save_every = int(dt_save // dt)
    n_save = int((t_total - t_burnin) // dt_save)
    result = np.empty((n_save, batch, 2), dtype=np.float32)

    p = np.zeros(batch)
    q = np.zeros(batch)

    # solve SDE
    i_save = 0
    for i in range(total_steps):
        f_conservative = -V_double_well(q, U0, x0, sigma)
        f_random = np.random.randn(batch) * np.sqrt(2 * dt * gamma / beta)
        f_dissipative = -gamma * p
        q += p * dt
        p += (f_conservative + f_dissipative) * dt + f_random
        if i * dt > t_burnin and i % save_every == 0:
            result[i_save, :, 0] = p
            result[i_save, :, 1] = q
            i_save += 1

    return result
