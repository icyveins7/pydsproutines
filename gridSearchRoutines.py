# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:47:58 2021

@author: Lken
"""

import numpy as np
import scipy as sp
import cupy as cp
import time

def gridSearchTDOA(s1x_list, s2x_list, tdoa_list, td_sigma_list, xrange, yrange, z, verb=True):
    '''
    Assumes a flat surface.
    z: height of surface
    '''
    xm, ym = np.meshgrid(xrange,yrange)
    fullmesh = np.vstack((xm.flatten(),ym.flatten(),np.zeros(len(ym.flatten())) + z)).transpose().astype(np.float32)
    cost_grid = None
    
    t1g = time.time()
    for i in range(len(tdoa_list)):
        # cpu code
        s1x = s1x_list[i].astype(np.float32)
        s2x = s2x_list[i].astype(np.float32)
        tdoa = tdoa_list[i].astype(np.float32)
        td_sigma = td_sigma_list[i].astype(np.float32)
        
        r = np.float32(tdoa * 299792458.0)
        r_sigma = np.float32(td_sigma * 299792458.0)
        
        rm = np.linalg.norm(s2x - fullmesh, axis=1) - np.linalg.norm(s1x - fullmesh, axis=1)
        
        if cost_grid is None:
            cost_grid = (r - rm)**2 / r_sigma**2
        else:
            cost_grid = cost_grid + (r - rm)**2 / r_sigma**2
        
    t2g = time.time()
    if verb:
        print("Grid search took %g seconds." % (t2g-t1g))
    
    return cost_grid

gridsearchtdoa_kernel = cp.RawKernel(r'''
extern "C" __global__
void gridsearchtdoa_kernel(int len, float *s1x_list, float *s2x_list,
                           float *tdoa_list, float *td_sigma_list,
                           float x0, float y0, float xp, float yp,
                           int xn, int yn, float z,
                           float *cost_grid)
{
    // allocate shared memory for the fm_slice
    extern __shared__ float s[];
    float *s_s1x_l = s; // (len * 3) floats
    float *s_s2x_l = (float*)&s_s1x_l[len * 3]; // (len * 3) floats
    float *s_r_l = (float*)&s_s2x_l[len * 3]; // (len) floats
    float *s_rsigma_l = (float*)&s_r_l[len]; // (len) floats
    
    // load shared memory
    for (int t = threadIdx.x; t < len * 3; t = t + blockDim.x){
        s_s1x_l[t] = s1x_list[t];
        s_s2x_l[t] = s2x_list[t];
    }
    for (int t = threadIdx.x; t < len; t = t + blockDim.x){
        s_r_l[t] = tdoa_list[t] * 299792458.0;
        s_rsigma_l[t] = td_sigma_list[t] * 299792458.0; // perform the multiplies while loading
    }
    
    __syncthreads();
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / xn;
    int col = tid % xn;
    
    float x = col * xp + x0;
    float y = row * yp + y0;
    
    float rm, rm2, rm1;
    float fullcost = 0.0f;
    float cost;
    
    for (int i = 0; i < len; i++){
        
        rm1 = norm3df(s_s1x_l[i*3+0] - x, s_s1x_l[i*3+1] - y, s_s1x_l[i*3+2] - z);
        rm2 = norm3df(s_s2x_l[i*3+0] - x, s_s2x_l[i*3+1] - y, s_s2x_l[i*3+2] - z);
        rm = rm2 - rm1; // theoretical range for the point
        
        cost = (s_r_l[i] - rm) / s_rsigma_l[i];
 
        fullcost = fmaf(cost, cost, fullcost); // accumulate costs (len) times
        
    }
    
    // write to output
    int cost_grid_len = xn * yn;
    if (tid < cost_grid_len){
        cost_grid[tid] = fullcost;
    }
}
''', 'gridsearchtdoa_kernel')

def gridSearchTDOA_gpu(s1x_list, s2x_list, tdoa_list, td_sigma_list, xrange, yrange, z, verb=True, moveToCPU=False):
    d_s1x_l = cp.asarray(s1x_list).astype(cp.float32)
    d_s2x_l = cp.asarray(s2x_list).astype(cp.float32)
    d_tdoa_l = cp.asarray(tdoa_list).astype(cp.float32)
    d_tdsigma_l = cp.asarray(td_sigma_list).astype(cp.float32)
    
    x0 = np.min(xrange).astype(np.float32)
    xp = (xrange[1]-xrange[0]).astype(np.float32)
    xn = len(xrange)
    y0 = np.min(yrange).astype(np.float32)
    yp = (yrange[1]-yrange[0]).astype(np.float32)
    yn = len(yrange)
    
    # prepare output
    d_cost_grid = cp.zeros(xn*yn, dtype=cp.float32)
    
    # run kernel
    t1 = time.time()
    THREADS_PER_BLOCK = 128
    NUM_BLOCKS = int(d_cost_grid.size/THREADS_PER_BLOCK + 1)
    gridsearchtdoa_kernel((NUM_BLOCKS,),(THREADS_PER_BLOCK,), (len(s1x_list), d_s1x_l, d_s2x_l, d_tdoa_l, d_tdsigma_l,
                                                               x0, y0, xp, yp, xn, yn, z, d_cost_grid), 
                          shared_mem=(d_s1x_l.size + d_s2x_l.size + d_tdoa_l.size + d_tdsigma_l.size) * 4)
    t2 = time.time()
    
    if verb:
        print("Grid search kernel took %g seconds." % (t2-t1))
    
    if moveToCPU:
        return cp.asnumpy(d_cost_grid)
    else:
        return d_cost_grid
