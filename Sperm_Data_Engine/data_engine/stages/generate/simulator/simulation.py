# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 05:22:11 2024

@author: cameron schmidt
"""

'''Sets up simulation env and agents'''

import numpy as np
import pandas as pd

def sample_param(spec, n):
    ''' return an (n, ) array from a scalar or distribution.
    currently supported: 
        {'dist':'normal','mean':μ,'sd':σ,'clip':[lo,hi]?}
        {'dist':'lognormal','mean_log':m,'sd_log':s,'clip':[lo,hi]?}
        {'dist':'uniform','low':a,'high':b}  
    '''
    
    if np.isscalar(spec):
        return np.full(n, float(spec), dtype= float)
    
    kind= spec.get('dist', 'normal')
    if kind == 'normal':
        x= np.random.normal(loc= spec['mean'], scale= spec['sd'], size= n)
        if 'clip' in spec: lo, hi= spec['clip']; x= np.clip(x, lo, hi)
        return x
    
    if kind == 'lognormal':
        x= np.random.lognormal(mean= spec['mean_log'], sigma= spec['sd_log'], size= n)
        if 'clip' in spec: lo, hi= spec['clip']; x= np.clip(x, lo, hi)
        return x
    
    if kind == 'uniform':
        return np.random.uniform(low= spec['low'], high= spec['high'], size= n)
    
    raise ValueError(f'[Warning] unknown distribution kind: {kind}')
        
        
class Simulation:
    def __init__(self, num_agents=3, num_steps= None, num_frames= 1000, frame_rate= 9, pixel_size=1.8, width=1024, height=1024, obs_agent_size=2, param_specs= None):
        # -- Environment parameters --
        self.num_agents = num_agents
        self.num_steps = num_steps # total number of frames in simulated microscopy video
        self.frame_rate = frame_rate # Frame rate from the original data for fitting. Used to generate same fps in output after downsampling.
        self.dt = 0.01 # Timestep [s]
        self.pixel_size = pixel_size # pixel size [um]/pixel
        self.steps_per_frame= max(1, int(round(1 / (self.frame_rate * self.dt))))
        # handle simulations by desired frame number
        if num_frames is not None:
            self.num_frames= int(num_frames)
            # ensure exact count after downsampling
            self.num_steps= 1 + (self.num_frames - 1) * self.steps_per_frame
            print(f'[Info] the number of steps in the simulation will be: {self.num_steps}')
        else: 
            # fallback to number of steps w/ 1000 step default
            self.num_steps= int(num_steps) if num_steps is not None else 1000
            self.num_frames= 1 + (self.num_steps - 1) // self.steps_per_frame
            print(f'[Info] the number of steps in the simulation will be: {self.num_steps}')
            
        self.width = width * self.pixel_size # Env width in pixels; width * pixel_size = um
        self.height = height * self.pixel_size
        
        # Agent size scaling parameters and calcs
        self.base_width = 1024 * self.pixel_size # Base x-dimension size for scaling
        self.base_height = 1024 * self.pixel_size
        self.obs_agent_size = obs_agent_size # diameter of agents from microscopy videos in pixels
        self.base_agent_size = obs_agent_size * self.pixel_size # Base agent size for scaling
        self.current_area = self.width * self.height
        self.base_area = self.base_width * self.base_height
        self.scale_factor = (self.base_area / self.current_area) ** 0.5 # Scales to the sqrt of the area ratio
        self.agent_size = self.base_agent_size * self.scale_factor # in um
        
        # Agent movement parameters (from param dists)
        self.v = 29.7 # nominal forward speed [um/s]
        self.OMEGA= .5 # slow chirality drift [rad/s]
        self.markov_prob= 0.001 # probability of sign change for OMEGA
        self.sigma_r = 0.0 # translational noise scale [um/sqrt(s)]
        self.sigma_theta = 0.1 #angular diffusion noise scale [rad/sqrt(s)]; Variance of the random increment is sigma squared dt
        
        # short timescale 'wobble' movement parameters
        self.wobble_w= 2 * np.pi * 3.0 # [rad/s]; 2 * pi * freq (Hz); represents the short timescale frequency
        self.wobble_delta= 1.0 # [um]; set > 0 to enable wobble (e.g., 0.06); represents the short timescalte amplitude
        
        specs= {
            'v': 29.7,  # [um/s]
            'OMEGA': 0.5,   # [rad/s]
            'sigma_theta': 0.1,     # [rad/sqrt(s)]
            'sigma_r': 0.0,     # [um/sqrt(s)]
            'wobble_w': 2 * np.pi * 3.0,    # [rad/s]; 2 * pi * freq (Hz)
            'wobble_delta': 1.0,    # [um]
            'wobble_phi': {'dist': 'uniform', 'low': 0.0, 'high': 2 * np.pi} # [rad]
            }
        
        if param_specs is not None: 
            specs.update(param_specs)
            
        n= self.num_agents
        self.v= sample_param(specs['v'], n)
        self.OMEGA= sample_param(specs['OMEGA'], n)
        self.sigma_theta= sample_param(specs['sigma_theta'], n)
        self.sigma_r= sample_param(specs['sigma_r'], n)
        self.wobble_w= sample_param(specs['wobble_w'], n)
        self.wobble_delta= sample_param(specs['wobble_delta'], n)
        self.wobble_phi= sample_param(specs['wobble_phi'], n)
        
        # probability of sign change for OMEGA
        self.markov_prob= 0.001
        
        # Initialize positions and orientations
        self.positions = np.zeros((self.num_steps, self.num_agents, 2)) # Stores the positions as an array of agent coordinates
        self.theta = np.random.rand(self.num_agents) * 2 * np.pi # initialize directions
        self.s_t= np.ones(self.num_agents, dtype=int) # Initialize the markov states for OMEGA
        self.initialize_positions()
        
    def initialize_positions(self):
        '''initialize the agend positions randomly within the env'''
        self.positions[0, :, 0] = np.random.uniform(0, self.width, self.num_agents)
        self.positions[0, :, 1] = np.random.uniform(0, self.height, self.num_agents)
        
    def run_simulation(self):
        '''run the main simulation loop to update agent positions'''
        for t in range(1, self.num_steps):
            self.update_positions(t)
            
    def update_positions(self, t): 
        '''update agent positons using Maciej Majkas movement functions'''
        dt= self.dt
        time_s= t * dt # physical time in seconds
        
        # identify active agents (with valid theta values)
        active_agents= ~np.isnan(self.theta)
        
        # generate noise for active agents
        eta_r= np.zeros(self.num_agents)
        eta_theta= np.zeros(self.num_agents)
        n_active= active_agents.sum()
        if n_active > 0:
            eta_r[active_agents] = np.random.normal(0, 1.0, n_active) # generate noise from a Gaussian distribution
            eta_theta[active_agents] = np.random.normal(0, 1.0, n_active)
        
        # Update s(t) using Markov process; sets the sign of the long timescale angular velocity Omega
        transition_mask= np.random.rand(self.num_agents) < self.markov_prob # decide which agents transition
        self.s_t[transition_mask] *= -1 # Transition s(t) for selected agents
                    
        # angular diffusion with sqrt(dt) scaling
        dtheta_noise= self.sigma_theta * np.sqrt(dt) * eta_theta
        self.theta[active_agents] += (self.s_t[active_agents] * self.OMEGA[active_agents] * dt + dtheta_noise[active_agents])
        
        # velocity with long timescale chirality
        cth= np.cos(self.theta[active_agents])
        sth= np.sin(self.theta[active_agents])
        
        # nominal foward speed
        v_par= self.v[active_agents] # [um/s]
        
        # small lateral wobble
        mod= self.wobble_w[active_agents] * self.wobble_delta[active_agents] * np.cos(self.wobble_w[active_agents] * time_s + self.wobble_phi[active_agents]) # [um/s]
        
        vx= v_par * cth - mod * sth
        vy= v_par * sth + mod * cth
        
        # translational diffusion
        # DATA-ENGINE FIX: this was multiplied by a hardcoded 0.0 ("initialize
        # as zeros") in the original -- sigma_r had zero effect on simulated
        # motion regardless of its configured value. Removed so fitting
        # sigma_r from real data (data_engine/stages/generate/fit_params.py)
        # is actually meaningful. Noise is applied along each agent's current
        # heading (cth/sth), not as isotropic 2D noise -- a simplified model
        # choice preserved as-is from the original formula structure.
        dx_diff= self.sigma_r[active_agents] * np.sqrt(dt) * eta_r[active_agents] * cth
        dy_diff= self.sigma_r[active_agents] * np.sqrt(dt) * eta_r[active_agents] * sth
        
        # position update
        dx= vx * dt + dx_diff
        dy= vy * dt + dy_diff
        
        
        self.positions[t, active_agents, 0] = self.positions[t-1, active_agents, 0] + dx
        self.positions[t, active_agents, 1] = self.positions[t-1, active_agents, 1] + dy
        
        
        # Detect and handle boundary crossings
        for i in range(self.num_agents):
            if np.isnan(self.theta[i]):
                continue # skip inactive agents
            if (self.positions[t, i, 0] < 0 or self.positions[t, i, 0] > self.width or self.positions[t, i, 1] < 0 or self.positions[t, i, 1] > self.height):
                # Assign a new agent_id and reset position
                self.positions[t, i, 0] %= self.width
                self.positions[t, i, 1] %= self.height
                self.split_trajectory(i, t)
                
    def split_trajectory(self, agent_index, t): 
        '''split trajectory of an agent that crosses the x or y boundaries'''
        ''' stop the trajectory of the original agent'''
        
        new_positions = np.zeros((self.num_steps, 2)) # initialize new trajectory array
        new_positions[:t, :]= np.nan #set previous positions to NaN
        new_positions[t:, :]= self.positions[t:, agent_index, :] #write new positions
        self.positions= np.append(self.positions, new_positions[:, np.newaxis, :], axis=1) # add the new agent
        self.num_agents+= 1 # increment the total agents
        self.s_t= np.append(self.s_t, np.random.choice([-1,1])) # initialize the new agent's s(t)
        
        # Update theta by adding a new random orientation 
        new_theta= np.random.rand() * 2 * np.pi
        self.theta= np.append(self.theta, new_theta)
        
        # copy each agents parameters to new trajectory
        self.v= np.append(self.v, self.v[agent_index])
        self.OMEGA= np.append(self.OMEGA, self.OMEGA[agent_index])
        self.sigma_theta= np.append(self.sigma_theta, self.sigma_theta[agent_index])
        self.sigma_r= np.append(self.sigma_r, self.sigma_r[agent_index])
        self.wobble_w= np.append(self.wobble_w, self.wobble_w[agent_index])
        self.wobble_delta= np.append(self.wobble_delta, self.wobble_delta[agent_index])
        self.wobble_phi= np.append(self.wobble_phi, self.wobble_phi[agent_index])
        
        # Deactivate the original agent by adding its future positions to NaN
        self.positions[t:, agent_index, :]= np.nan
        self.theta[agent_index]= np.nan
        
    def get_positions(self):
        '''return agent positions'''
        return self.positions
    
    def get_agent_size(self):
        '''return the calculated agent size'''
        return self.agent_size
    
    def save_trajectories(self, filename, frame_rate_out= None):
        '''write the trajectories out to a .csv file of 'filename' '''
        '''Downsamples to the input frame rate of the video, or to the desired output frame rate in fps'''
        
        # determine how often to sample and compute the sample indices
        if frame_rate_out is None: 
            frame_rate_out = self.frame_rate #default to configured frame rate in simulation
            steps_per_frame= self.steps_per_frame
            
        else: 
            steps_per_frame= max(1, int(round(1 / (frame_rate_out * self.dt))))
            
        sampled_indices= np.arange(0, self.num_steps, steps_per_frame)
        
        # downsample the saved positions using the above sample rate
        sampled_positions= self.positions[sampled_indices, :, :]
       
        
        # store the trajectories as arrays
        time= np.repeat(sampled_indices * self.dt, self.num_agents) # time (s)
        agent_id=np.tile(np.arange(self.num_agents), len(sampled_indices)) # tile(A, reps) repeats the entire array
        x=(sampled_positions[:, :, 0]).flatten() # .flatten() makes a 1D array from a 2D array
        y=(sampled_positions[:, :, 1]).flatten() # stores coords in [um]
        
        # filter out invalid trajectories (NaNs for split segments)
        valid_indices=~np.isnan(x) & ~np.isnan(y)
        
        # create the dataframe
        df=pd.DataFrame({
            'Time': time[valid_indices],
            'Agent_ID': agent_id[valid_indices],
            'x_coord': x[valid_indices],
            'y_coord': y[valid_indices],
            })
        
        # save to .csv format
        df.to_csv(filename, index=False)
        
        
        
        