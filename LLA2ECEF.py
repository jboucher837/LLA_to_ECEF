#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:46:06 2023

@author: jonathanboucher
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def main():
    # Read the CSV file
    df = pd.read_csv('SciTec_code_problem_data.csv')
    column_names = ['Time since the Unix epoch [seconds]', 'WGS84 latitude [degrees]', 'WGS84 longitude [degrees]', 'WGS84 altitude [kilometers]']
    df.columns = column_names

    # Constants calculations from LLAtoECEF.pdf
    a = 6378137
    b = 6356752.31424518
    e = np.sqrt((a**2-b**2)/a**2)
    a_sq = a**2
    b_sq = b**2
    e_sq = e**2
    
    # Convert LLA to ECEF
    #Latitude, Longitude, Altitude conversion from degrees to radians
    ##resource used for applying this package
    #https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html
    def LLA_to_ECEF(phi, lam, h):
        phi = np.deg2rad(phi)
        lam = np.deg2rad(lam)
        h = np.array(h).reshape((-1, 1))
        
        N = a/np.sqrt(1-e_sq*np.sin(phi)**2)
        X = (N+h)*np.cos(phi)*np.cos(lam)
        Y = (N+h)*np.cos(phi)*np.sin(lam)
        Z = ((b_sq/a_sq)*N+h)*np.sin(phi)
        
        return X, Y, Z
    
    # Create a new data frame for ECEF coordinates
    new_df = pd.DataFrame(columns=['X', 'Y', 'Z'])
    
    # Iterate over each row in the data frame
    for index, row in df.iterrows():
        lat = row['WGS84 latitude [degrees]']
        lon = row['WGS84 longitude [degrees]']
        #multiply by 1000 to convert from kilometers to meters
        alt = row['WGS84 altitude [kilometers]']*1000
        
        x, y, z = LLA_to_ECEF(lat, lon, alt)
        
        new_row = pd.DataFrame({'X': [x], 'Y': [y], 'Z': [z]}, index=[index])
        new_df = pd.concat([new_df, new_row])
    
    # Add time values to the new data frame
    time_values = df['Time since the Unix epoch [seconds]']
    new_df.insert(0, 'Time', time_values)
    
    # Calculate ECEF velocities
    def ECEFVelocity(df, t1, t2):
        df1 = df[df['Time'] == t1]
        df2 = df[df['Time'] == t2]
        
        tdiff = t2 - t1
        
        X1, X2 = df1['X'].values[0], df2['X'].values[0]
        Y1, Y2 = df1['Y'].values[0], df2['Y'].values[0]
        Z1, Z2 = df1['Z'].values[0], df2['Z'].values[0]
    
        del_X = X2 - X1
        del_Y = Y2 - Y1
        del_Z = Z2 - Z1
        
        vel_X = del_X / tdiff
        vel_Y = del_Y / tdiff
        vel_Z = del_Z / tdiff
        
        return vel_X, vel_Y, vel_Z
    
    # Initialize velocities in the new DataFrame
    new_df.at[0, 'Velocity_X'] = 0
    new_df.at[0, 'Velocity_Y'] = 0
    new_df.at[0, 'Velocity_Z'] = 0
    
    #https://stackoverflow.com/questions/19184335/is-there-a-need-for-rangelena
    #^This forum helped me develop this loop so that I could iterate through by checking the length
    
    #loop through i iterations for new_df-1, since its 500 observations and each calculation requires 2, we only need
    #to loop 499 times to cover each change
    
    # Calculate velocities for each time step
    for n in range(len(new_df) - 1):
        t1 = new_df['Time'].iloc[n]
        t2 = new_df['Time'].iloc[n + 1]
        vel_X, vel_Y, vel_Z = ECEFVelocity(new_df, t1, t2)
        
        new_df.at[n + 1, 'Velocity_X'] = vel_X
        new_df.at[n + 1, 'Velocity_Y'] = vel_Y
        new_df.at[n + 1, 'Velocity_Z'] = vel_Z
    
    # Interpolate velocities at specific times
    #https://docs.scipy.org/doc/scipy/tutorial/interpolate.html
    #python documentation on scipy interpolation packages, chose interp1d due to continuity issues
    def vel_interpolation(df, t):
        X = interp1d(df['Time'], df['Velocity_X'], kind='linear')
        Y = interp1d(df['Time'], df['Velocity_Y'], kind='linear')
        Z = interp1d(df['Time'], df['Velocity_Z'], kind='linear')
        
        int_X = X(t)
        int_Y = Y(t)
        int_Z = Z(t)
        
        print("Interpolated X, Y, and Z velocities at time", t, "=", int_X, int_Y, int_Z)
    
    # Interpolate velocities at specific times
    vel_interpolation(new_df, 1532334000)
    vel_interpolation(new_df, 1532335268)
    
    # Print the final data frame
    # print(new_df)


if __name__ == '__main__':
    main()