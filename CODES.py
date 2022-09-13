#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:49:19 2022

@author: abcd
"""

import math
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LinearRing

def coordinates_to_field(df): #(Map Projection)
    a = 6378.137
    e = 0.0818192
    k0 = 0.9996
    E0 = 500
    N0 = 0
    lon1 = []
    lat1 = []
    lons = df['longitude'].to_list()
    lats = df['latitude'].to_list()
    for i in range(len(df)):
        lon = lons[i]
        lat = lats[i]
        Zonenum = int(lon / 6) + 31
        lamda0 = (Zonenum - 1) * 6 - 180 + 3
        lamda0 = lamda0 * math.pi / 180
        phi = lat * math.pi / 180
        lamda = lon * math.pi / 180
        v = 1 / math.sqrt(1 - e ** 2 * math.sin(phi) ** 2)
        A = (lamda - lamda0) * math.cos(phi)
        T = math.tan(phi) ** 2
        C = e ** 2 * math.cos(phi) * math.cos(phi) / (1 - e ** 2)
        s = (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * phi - \
            (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * math.sin(2 * phi) + \
            (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * math.sin(4 * phi) - \
            35 * e ** 6 / 3072 * math.sin(6 * phi)
        UTME = E0 + k0 * a * v * (A + (1 - T + C)*A ** 3 / 6+(5 - 18 * T + T ** 2) * A ** 5 / 120)
        UTMN = N0 + k0 * a * (s + v * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 + (61 - 58 * T + T ** 2) * A ** 6 / 720))
        # UTME,UTMN (Kilometres). Converting them into Meters
        UTME = UTME * 1000
        UTMN = UTMN * 1000
        lat1.append(UTME)
        lon1.append(UTMN)
    print (lon1)
    print (lat1)
    return pd.DataFrame({'X': lon1
                        , 'Y': lat1})

def VexRotated (vertex, RM):
    rotation = np.dot(vertex, RM)
    return rotation[0,0], rotation[0,1]

def read_match_data(MATCHDATADIR):
    # read match (SSG) match information: Date, Category, Format, Team in SSGs, Player Name, Split Start Time, Split End Time
    match = pd.read_excel(MATCHDATADIR, sheet_name = 1, usecols = ['Date', 'Category', 'Format','Number of team ', 'Player Name', 'Split Start Time', 'Split End Time'])
    return match

def TeamTracking(file_dir, teamname, StartTS, EndTS, RM):
    file_list = os.listdir(os.path.join(file_dir, teamname))
    print (f"{file_list}\n") # check if there are only player positional data files
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    
    # Create a DataFrame for later use
    TeamPosition = pd.DataFrame()
    
    for file in file_list:
        print (file)
        path = os.path.join(file_dir, teamname, file)
        position = pd.read_csv(path, usecols = [0, 2, 3]) # read useful columns
        position['Excel Timestamp'] = position['Excel Timestamp'].round(6) # round to floats with six decimals
        
        if len(position.loc[position['Excel Timestamp'] == StartTS]) >= 1: # start timestamp is found
            StartIndex = position.loc[position['Excel Timestamp'] == StartTS].index[0] # Getting StartIndex from position data
            print (f"StartIndex found at first time: {StartIndex}")
        else:
            for i in range (1, 10):
                print (f"i = {i}")
                if len(position.loc[(position['Excel Timestamp'] * 1000000).astype(int) == int(StartTS*1000000)+i]) >= 1: # if data at StartTS got lost, then start from next TS
                    print (f"i = {i}")
                    StartIndex = position.loc[(position['Excel Timestamp'] * 1000000).astype(int) == int(StartTS*1000000)+i].index[0]
                    print ("StartIndex")
                    print (StartIndex)
                    break
                else:
                    print (f"StartIndex {StartTS+i*0.000001} Not Found")
                    
        if len(position.loc[position['Excel Timestamp'] == EndTS]) >= 1: # end timestamp is found
            EndIndex = position.loc[position['Excel Timestamp'] == EndTS].index[-1] # Getting EndIndex from position data
            print (f"EndIndex found at first time: {EndIndex}")
        else:
            for i in range (1, 10):
                print (f"i = {i}")
                if len(position.loc[(position['Excel Timestamp'] * 1000000).astype(int) == int(EndTS*1000000)-i]) >= 1: # if data at EndTS got lost, the end at last TS
                    print (f"i = {i}")
                    EndIndex = position.loc[(position['Excel Timestamp'] * 1000000).astype(int) == int(EndTS*1000000)-i].index[-1]
                    print ("EndIndex")
                    print (EndIndex)
                    break
                else:
                    print (f"EndIndex {EndTS+i*0.000001} Not Found")
        
        position = position.iloc[StartIndex:EndIndex+1,:] # Subsetting by StartIndex and EndIndex
        print (f"StartIndex: {StartIndex}")
        print (f"EndIndex: {EndIndex}")
        print (position)
        
        ### Lat & Lon to X & Y (Map Projection)
        a = 6378.137
        e = 0.0818192
        k0 = 0.9996
        E0 = 500
        N0 = 0
        lon1 = []
        lat1 = []
        lons = position[' Longitude'].to_list()
        lats = position[' Latitude'].to_list()
        for k in range(len(position)):
            lon = lons[k]
            lat = lats[k]
            Zonenum = int(lon / 6) + 31
            lamda0 = (Zonenum - 1) * 6 - 180 + 3
            lamda0 = lamda0 * math.pi / 180
            phi = lat * math.pi / 180
            lamda = lon * math.pi / 180
            v = 1 / math.sqrt(1 - e ** 2 * math.sin(phi) ** 2)
            A = (lamda - lamda0) * math.cos(phi)
            T = math.tan(phi) ** 2
            C = e ** 2 * math.cos(phi) * math.cos(phi) / (1 - e ** 2)
            s = (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * phi - \
                (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * math.sin(2 * phi) + \
                (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * math.sin(4 * phi) - \
                35 * e ** 6 / 3072 * math.sin(6 * phi)
            UTME = E0 + k0 * a * v * (A + (1 - T + C)*A ** 3 / 6+(5 - 18 * T + T ** 2) * A ** 5 / 120)
            UTMN = N0 + k0 * a * (s + v * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 + (61 - 58 * T + T ** 2) * A ** 6 / 720))
            # UTME,UTMN based on Kilometres. Converting them into Meters
            UTME = UTME * 1000
            UTMN = UTMN * 1000
            lat1.append(UTME)
            lon1.append(UTMN)
             #print (lat1)
             #print (lon1)
        position["X"] = lon1
        position["Y"] = lat1
        #print (position)
        
        ### apply RM to X & Y
        for i in range (len(position)):
            pos = position.iloc[[i], [3,4]]
            position.iloc[[i], [3, 4]] = np.dot(pos, RM)
        
        ### switch X and Y? Depends on whether pitch x,y were switched; Referring to line 259
        #position[["X", "Y"]] = position[["Y", "X"]]
        
        ### Whether each Timestamp is unique?
        if len(position["Excel Timestamp"].unique()) != len(position):
            print ("!!! Same Timestamp occurs !!!")
        
        ### drop lan & lon
        position.drop(columns=[" Latitude", " Longitude"], inplace = True)
        
        ### amend column name
        playername = file[12:16]
        position.columns = ["Timestamp", "{}_x".format(playername), "{}_y".format(playername)]
        
        ### merging players in the same team together (full outer join)
        if file_list.index(file) == 0:
            TeamPosition = position
        else:
            TeamPosition = pd.merge(TeamPosition, position, on = 'Timestamp', how = 'outer')
    
    ### Modifying column name to "PLAYERNAME_x" or "PLAYERNAME_y"
    TeamPosition.sort_values(by = 'Timestamp', axis=0, ascending = True, inplace = True)
    
    ### reset Frame
    TeamPosition.reset_index(drop = True)
    print (TeamPosition)
    return TeamPosition



### 1) reading match info, selecting useful columns

MATCHDATADIR = '/... FILE PATH of MATCH INFO .../SSGINFO_6v6_32x24.xlsx'
match_info = read_match_data(MATCHDATADIR)

### 2) reading pitch info

df = pd.read_excel("/... FILE PATH of PITCH INFO .../Spanish Academy Pitch2.xlsx")


### 3) Covert Lat & Lon to 2D Coordinates, and plot

ini_xyco_pitch = coordinates_to_field(df) # get coordinates(x, y) of pitch
ini_pitch_x, ini_pitch_y = LinearRing(zip(ini_xyco_pitch['X'], ini_xyco_pitch['Y'])).xy # plot the pitch (explicitly closed polygon)
fig = plt.figure()
plt.plot(ini_pitch_x, ini_pitch_y, "g", alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


### 4) Rotation

""" Rotation Matrix """

'''
1) find the Origin, left below one of 4 vertices
2) find the other vertex that should be on x-axis
2) calculate the angle
3) matrix
4) apply into other vertices
'''

### Origin and the other vertex
ini_xyco_pitch.sort_values(by = 'Y', axis=0, ascending = True, inplace = True)
Origin = ini_xyco_pitch[0:2].sort_values(by = 'X', axis=0, ascending=True).iloc[[0]]
print (Origin)
TheOther = ini_xyco_pitch[0:2].sort_values(by = 'X', axis=0, ascending=True).iloc[[-1]] # the other vertex on x-axis
print (TheOther)
ThirdVex = ini_xyco_pitch[2:3]
print (ThirdVex)
FourthVex = ini_xyco_pitch[3:4]
print (FourthVex)

### Calculate the angle
dx = abs(float(TheOther['X']) - float(Origin['X']))
dy = abs(float(TheOther['Y']) - float(Origin['Y']))
angle = np.arctan2(dy, dx) * 180 / np.pi

### Rotation Matrix for clockwise rotating (RW_CW)
RM_CW = np.array([[np.cos(angle*np.pi/180), -np.sin(angle*np.pi/180)],
                  [np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])

### Rotation Matrix for counter-clockwise rotating (RW_CCW)
RM_CCW = np.array([[np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)],
                  [-np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])

### Clockwise or Counterclockwise rotating
if float(Origin['Y']) < float(TheOther['Y']):
    RotationMatrix = RM_CW
else:
    RotationMatrix = RM_CCW
    
PitchRotated = pd.DataFrame(columns = ['X', 'Y'])
for vex in (Origin, TheOther, ThirdVex, FourthVex): # apply rotation matrix to four vertices
    print (f"VERTEX: {vex}")
    PitchRotated.loc[len(PitchRotated)] = VexRotated(vex, RotationMatrix)
    print ("RESULT:", PitchRotated)
print (PitchRotated) # Get rotated pitch vextices

pitch_x, pitch_y = LinearRing(zip(PitchRotated['X'], PitchRotated['Y'])).xy
fig = plt.figure()
plt.plot(pitch_x, pitch_y, "g", alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

### If switching current X Y is needed? To make pitch length parallel with x-axis.
### If pitch length is parallel with with y-axis, it's needed.
### If X, Y are switched here, X and Y of player positional data should also be switched.
if PitchRotated['X'].max()-PitchRotated['X'].min() < PitchRotated['Y'].max()-PitchRotated['Y'].min() : 
    PitchRotated[["X","Y"]] = PitchRotated[["Y","X"]]
    print ("!! Pitch X,Y Switched !!")
else: print ("** Pitch X,Y Stay **")

pitch_x, pitch_y = LinearRing(zip(PitchRotated['X'], PitchRotated['Y'])).xy
fig = plt.figure()
plt.plot(pitch_x, pitch_y, "g", alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

""" --- Rotation Matrix END --- """


### 5) processing positional data

""" Processing positional data """

'''
1) Get the initial team position dataset
2) Fill in the NaN value
'''

### Getting Start & End TimeStamp from Match Data 
playernum = 6
StartTS = float(format(float(match_info.loc[0:playernum-1, ['Split Start Time']].max()), ".6f"))
EndTS = float(format(float(match_info.loc[0:playernum-1, ['Split End Time']].min()), ".6f"))
RM = RotationMatrix

### Folder Name
SSG = 'U18_6v6_32x24'

### Path to SSG positional data
file_dir = '/... FILE PATH of POSITIONAL DATA .../'

SSG6v6 = TeamTracking(file_dir, SSG, StartTS, EndTS, RM)
# Convert [Timestamp] into (int), for following merging
SSG6v6["Timestamp"] = SSG6v6["Timestamp"].map(lambda x: x*1000000).astype(int)
SSG6v6.reset_index(drop = True, inplace = True)

### Create a timeline of 10 Hz (int)
Time = pd.DataFrame({"Timestamp": list(range(int(StartTS*1000000), int(EndTS*1000000)+1, 1))})

### Create a new timeline start from 0.1s (10 Hz)
Time["Start [s]"] = Time.index.map(lambda x: x*0.1)

## Count the number of rows with NaN value (not completely missing sampling), partly missing percent
print (f"Non-completely missed sampling of {SSG}: {len(SSG6v6[SSG6v6.isnull().T.any()])},\n Percent: {len(SSG6v6[SSG6v6.isnull().T.any()]) / len(Time)}")

## How many rows of missed data (completely missing sampling)
print (f"missed sampling of {SSG}: {len(Time) - len(SSG6v6)},\n Percent: {(len(Time) - len(SSG6v6)) / len(Time)}")

## Count how many continuous NaNs
NaNs = []
for player in SSG6v6.columns[[1, 3, 5, 7, 9, 11]]:
    print (player)
    LIST = list(SSG6v6[SSG6v6[player].isnull()].index)
    for i in LIST:
        #print (i)
        if LIST.index(i) == len(LIST) - 1:
            break
        elif LIST.index(i) != len(LIST) - 1 and i+1 == LIST[LIST.index(i)+1]:
            NaNs.append("2 NaNs")
            print ("!2 NaNs found!")
            if LIST.index(i) != len(LIST) - 2 and i+2 == LIST[LIST.index(i)+2]:
                NaNs.append("3 NaNs")
                print ("!!3 NaNs found!!")
                if LIST.index(i) != len(LIST) - 3 and i+3 == LIST[LIST.index(i)+3]:
                    NaNs.append("4 NaNs")
                    print ("!!!4 NaNs found!!!")
                    if LIST.index(i) != len(LIST) - 4 and i+4 == LIST[LIST.index(i)+4]:
                        NaNs.append("5 NaNs")
                        print (f"!!!!5 NaNs found!!!! index = {i}")
                        if LIST.index(i) != len(LIST) - 5 and i+5 == LIST[LIST.index(i)+5]:
                            NaNs.append("6 NaNs")
                            print (f"!!!!6 NaNs found!!!! index = {i}")
# 2 continuous NaNs
print ("2 continuous NaNs:", NaNs.count("2 NaNs") - NaNs.count("3 NaNs"))

# 3 continuous NaNs
print ("3 continuous NaNs", NaNs.count("3 NaNs") - NaNs.count("4 NaNs"))

# 4 continuous NaNs
print ("4 continuous NaNs", NaNs.count("4 NaNs") - NaNs.count("5 NaNs"))

# 5 continuous NaNs
print ("5 continuous NaNs", NaNs.count("5 NaNs") - NaNs.count("6 NaNs"))


## Merging with new timeline
SSG6v6_10Hz = pd.merge(Time, SSG6v6, on = "Timestamp", how = "outer")

## Interpolation
SSG6v6_10Hz.interpolate(method="linear", limit_direction="forward", inplace=True, axis=0)

## Preprocessing end, output
SSG6v6_10Hz.to_csv("{}_10Hz.csv".format(SSG), index=0, na_rep="NA")

""" --- Processing Positional Data END --- """
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    