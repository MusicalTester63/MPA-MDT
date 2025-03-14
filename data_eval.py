# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:57:21 2020

@author: Dell
"""
import json
import os
from random import random
from scipy.optimize import curve_fit,minimize
# from lmfit import minimize, Parameters, report_fit, Model
import astropy.units as u
import numpy as np
from numpy import cos, sin, tan, degrees, radians, arcsin, pi, arccos, array
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, Angle
import matplotlib.pyplot as plt
from point_eqs import *


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

plt.rcParams.update({'font.size': 14})

s = 0.0001
n = 0
def lt(pos):
    return [[x[0] for x in pos],[x[1] for x in pos]]
def lt1(pos):
    return [[x[0][0] for x in pos],[x[0][1] for x in pos]]
    

def point(t, dec, zh, zd, ma, me, mp, tf, co, df, fo):
    phi = EarthLocation(lat=48.3733*u.deg,lon=17.240*u.deg,height=531.1*u.m).lat.rad
    tc = t + zh + co/(cos(dec)) + mp*tan(dec) - ma*cos(t)*tan(dec) + me*sin(t)*tan(dec) + tf*cos(phi)*sin(t)/cos(dec) - df*(cos(phi)*cos(t) + sin(phi)*tan(dec))
    decc = dec + zd + ma*sin(t) + me*cos(t) + tf*(cos(phi)*cos(t)*sin(dec) - sin(phi)*cos(dec)) + fo*cos(t)
    
    return array([tc,decc])

def rmsus(meas_pos, pos_ast):
    # Vypočítaj RMS (Root Mean Square) medzi dvoma polohami
    return np.sqrt(np.mean((meas_pos - pos_ast) ** 2))
    
def get_model_data(mode):
    with open('./data/data.txt','r') as f:
        text=f.readlines()[1:]
        n=0
        if mode == '1':
            meas_pos = [[],[]]
            pos_ast = [[],[]]
            for line in text:
                if line != '':
                    meas_pos[0].append(radians(float(line.split('\t')[2])))
                    meas_pos[1].append(radians(float(line.split('\t')[3])))
                    pos_ast[0].append(radians(float(line.split('\t')[0])))
                    pos_ast[1].append(radians(float(line.split('\t')[1])))
                    n+=1                
        elif mode == '2':
            meas_pos = []
            pos_ast = []
            for line in text:
                meas_pos.append(radians(np.array([float(line.split('\t')[2]),float(line.split('\t')[3])])))
                pos_ast.append(radians(np.array([float(line.split('\t')[0]),float(line.split('\t')[1])])))
                n+=1

    return array(meas_pos), array(pos_ast), array(meas_pos).shape
meas_pos, pos_ast, n = get_model_data('1')

def metric(par,*args):
    data = list(get_model_data('1'))
    
    
    if args != ():
        # print(args[0],args[1])
        data[0]=data[0][0:2,args[0]:args[1]:1]
        data[1]=data[1][0:2,args[0]:args[1]:1]    
    # print(data[0],data[1])
    calcul = point(t=data[0][0], dec=data[0][1], zh=par[0], zd=par[1], ma=par[2], me=par[3], mp=par[4], tf=par[5], co=par[6], df=par[7], fo=par[8])
    
    scalar = np.sum(((calcul[0]-data[1][0])/0.0001)**2+((calcul[1]-data[1][1])/0.0001)**2)

    return scalar




#print(n[1]//10)

lst = [i+1 for i in range(0,n[1])]
steps = [lst[i:i + 15] for i in range(0, len(lst), 15)]
steps = [[steps[i][0],steps[i][-1]] for i in range(len(steps))]
steps[0][0] = 0
print("Coordinates has been read.")
for k in range(len(steps)):
    if k == 0:
        x0=array([0,0,0,0,0,0,0,0,0])
        resultat = minimize(metric,x0,args=(0,steps[k][1]),method='nelder-mead',options={'xatol': 1e-7,'maxiter':10000, 'disp': False})
        # print(degrees(resultat.x),'deg')
        params = list(resultat.x)
        pos_f = point(meas_pos[0],meas_pos[1],*params)
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        print('RMS-without PM:',rmsus(meas_pos,pos_ast)*3600,'"')
        print(f'After {k} iterations, acounting {steps[k][1]} points')
        print('Current RMS-with PM:',rmsus(pos_f,pos_ast)*3600,'"')
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        pos_plot = meas_pos
        pos_ch_plot = pos_ast
        pos_f_plot = pos_f
        legend_elements = [Line2D([0], [0], marker='>', color='g', label=f'Point-Ast\nRMS={np.round(rmsus(pos_f,pos_ast)*3600,3)}"'),
                           Line2D([0], [0], marker='o', color='w', label='0.5deg',
                                  markeredgecolor='r',markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='0.25deg',
                                  markeredgecolor='g',markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='0.05deg',
                                  markeredgecolor='b',markerfacecolor='w', markersize=15)]
        
        circle1 = plt.Circle((0, 0), 0.5,color='r', fill=False,label='0.5deg')
        circle2 = plt.Circle((0, 0), 0.25, color='g', fill=False,label='0.25deg')
        circle3 = plt.Circle((0, 0), 0.05, color='b', fill=False,label='0.05deg')
        
        
        fig,(ax1,ax2)=plt.subplots(1,2, figsize=(16,8))
        ax2.axhline(y=0, color="black")
        ax2.axvline(x=0, color="black")
        
        fig.suptitle(f'Apparent encoder possition with and without pointing model versus astrometric position - N = {steps[k][1]}')
        for i in range(len(pos_plot[0])):
            ax2.plot(degrees(pos_f_plot[0][i])-degrees(pos_ch_plot[0][i]),degrees(pos_f_plot[1][i])-degrees(pos_ch_plot[1][i]),'g>')
        
        ax2.set_xlabel('Post-pointing hour angle distance [deg]')
        ax2.set_ylabel('Post-pointing declination distance [deg]')
        ax2.set_xlim(-0.5,0.5)
        ax2.set_ylim(-0.5,0.5)
        ax2.add_patch(circle1)
        ax2.add_patch(circle2)
        ax2.add_patch(circle3)
        ax2.legend(handles=legend_elements, loc='best')
        legend_elements = [Line2D([0], [0], marker='.', color='r', label=f'WithoutPoint-Ast\nRMS={np.round(rmsus(meas_pos,pos_ast)*3600,3)}"'),
                           Line2D([0], [0], marker='o', color='w', label='0.5deg',
                                  markeredgecolor='r',markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='0.25deg',
                                  markeredgecolor='g',markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='0.05deg',
                                  markeredgecolor='b',markerfacecolor='w', markersize=15)]
        
        circle1 = plt.Circle((0, 0), 0.5, color='r', fill=False,label='0.5deg')
        circle2 = plt.Circle((0, 0), 0.25, color='g', fill=False,label='0.25deg')
        circle3 = plt.Circle((0, 0), 0.05, color='b', fill=False,label='0.05deg')
        ax1.axhline(y=0, color="black")
        ax1.axvline(x=0, color="black")
        for i in range(len(pos_plot[1])):
            ax1.plot(degrees(pos_plot[0][i])-degrees(pos_ch_plot[0][i]),degrees(pos_plot[1][i])-degrees(pos_ch_plot[1][i]),'r.')
            # ax2.plot(i,degrees(pos_ch_plot[1][i]),'bx')
        ax1.set_xlabel('Pre-pointing hour angle distance [deg]')
        ax1.set_ylabel('Pre-pointing declination distance [deg]')
        ax1.set_xlim(-0.5,0.5)
        ax1.set_ylim(-0.5,0.5)
        ax1.add_patch(circle1)
        ax1.add_patch(circle2)
        ax1.add_patch(circle3)
        
        
        ax1.legend(handles=legend_elements, loc='best')
        plt.savefig(f'PM-iter-{k}-part.png', format='png',dpi=600)
        plt.show()
        
    else:
        x0=array(params)
        resultat = minimize(metric,x0,args=(0,steps[k][1]),method='nelder-mead',options={'xatol': 1e-7,'maxiter':10000, 'disp': False})
        # print(degrees(resultat.x),'deg')
        params = list(resultat.x)
        #print(resultat.fun)
        pos_f = point(meas_pos[0],meas_pos[1],*params)
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        print('RMS-without PM:',rmsus(meas_pos,pos_ast)*3600,'"')
        print(f'After {k} iterations, acounting {steps[k][1]} points')
        print('Current RMS-with PM:',rmsus(pos_f,pos_ast)*3600,'"')
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')

        pos_plot = meas_pos
        pos_ch_plot = pos_ast
        pos_f_plot = pos_f
        legend_elements = [Line2D([0], [0], marker='>', color='g', label=f'Point-Ast\nRMS={np.round(rmsus(pos_f,pos_ast)*3600,3)}"'),
                           Line2D([0], [0], marker='o', color='w', label='0.5deg',
                                  markeredgecolor='r',markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='0.25deg',
                                  markeredgecolor='g',markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='0.05deg',
                                  markeredgecolor='b',markerfacecolor='w', markersize=15)]
        
        circle1 = plt.Circle((0, 0), 0.5, color='r', fill=False,label='0.5deg')
        circle2 = plt.Circle((0, 0), 0.25, color='g', fill=False,label='0.25deg')
        circle3 = plt.Circle((0, 0), 0.05, color='b', fill=False,label='0.05deg')
        
        
        fig,(ax1,ax2)=plt.subplots(1,2, figsize=(16,8))
        ax2.axhline(y=0, color="black")
        ax2.axvline(x=0, color="black")
        
        fig.suptitle(f'Apparent encoder possition with and without pointing model versus astrometric position - N = {steps[k][1]}')
        for i in range(len(pos_plot[0])):
            ax2.plot(degrees(pos_f_plot[0][i])-degrees(pos_ch_plot[0][i]),degrees(pos_f_plot[1][i])-degrees(pos_ch_plot[1][i]),'g>')
        
        ax2.set_xlabel('Post-pointing hour angle distance [deg]')
        ax2.set_ylabel('Post-pointing declination distance [deg]')
        ax2.set_xlim(-0.5,0.5)
        ax2.set_ylim(-0.5,0.5)
        ax2.add_patch(circle1)
        ax2.add_patch(circle2)
        ax2.add_patch(circle3)
        ax2.legend(handles=legend_elements, loc='best')
        legend_elements = [Line2D([0], [0], marker='.', color='r', label=f'WithoutPoint-Ast\nRMS={np.round(rmsus(meas_pos,pos_ast)*3600,3)}"'),
                           Line2D([0], [0], marker='o', color='w', label='0.5deg',
                                  markeredgecolor='r',markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='0.25deg',
                                  markeredgecolor='g',markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='0.05deg',
                                  markeredgecolor='b',markerfacecolor='w', markersize=15)]
        
        circle1 = plt.Circle((0, 0), 0.5, color='r', fill=False,label='0.5deg')
        circle2 = plt.Circle((0, 0), 0.25, color='g', fill=False,label='0.25deg')
        circle3 = plt.Circle((0, 0), 0.05, color='b', fill=False,label='0.05deg')
        ax1.axhline(y=0, color="black")
        ax1.axvline(x=0, color="black")
        for i in range(len(pos_plot[1])):
            ax1.plot(degrees(pos_plot[0][i])-degrees(pos_ch_plot[0][i]),degrees(pos_plot[1][i])-degrees(pos_ch_plot[1][i]),'r.')
            # ax2.plot(i,degrees(pos_ch_plot[1][i]),'bx')
        ax1.set_xlabel('Pre-pointing hour angle distance [deg]')
        ax1.set_ylabel('Pre-pointing declination distance [deg]')
        ax1.set_xlim(-0.5,0.5)
        ax1.set_ylim(-0.5,0.5)
        ax1.add_patch(circle1)
        ax1.add_patch(circle2)
        ax1.add_patch(circle3)
        
        
        ax1.legend(handles=legend_elements, loc='best')
        plt.savefig(f'PM-iter-{k}-part.png', format='png',dpi=600)
        
        plt.show()
        


with open('AGO70_params.json','w') as js:
    names=['zh', 'zd', 'ma', 'me', 'mp', 'tf', 'co', 'df', 'fo']
    json.dump({key:params[i] for i,key in enumerate(names)},js,indent=3)
    js.close()
    if os.path.exists('AGO70_params.json'):
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        print('Params was saved -> AGO70_params.json')
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+')
        print('Name \t Value')
        for i,j in enumerate(names):
            print(j,'\t',np.round(degrees(params[i])*3600,3),'arc-sec')
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+')


pos_f = point(meas_pos[0],meas_pos[1],*params)


pos_plot = meas_pos
pos_ch_plot = pos_ast
pos_f_plot = pos_f
       


c=SkyCoord(ra=pos_plot[0], dec=pos_plot[1], unit=(u.rad, u.rad))
ra_rad = c.ra.radian
dec_rad = c.dec.radian
c1=SkyCoord(ra=pos_ch_plot[0], dec=pos_ch_plot[1], unit=(u.rad, u.rad))
ra_rad1 = c1.ra.radian
dec_rad1 = c1.dec.radian
c2=SkyCoord(ra=pos_f_plot[0], dec=pos_f_plot[1], unit=(u.rad, u.rad))
ra_rad2 = c2.ra.radian
dec_rad2 = c2.dec.radian

fig = plt.figure()
legend_elements = [Line2D([0], [0], marker='.', color='r', label='Ast, pos.'),Line2D([0], [0], marker='>', color='b', label='without point.'),Line2D([0], [0], marker='x', color='g', label='With point')]

fig.suptitle('Observational plan')
ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
# ax.invert_yaxis()
ax.set_theta_zero_location('N')
ax.set_rlim(-0.5, 1.6)  # dolná hranica, horná hranica
ax.plot(ra_rad,dec_rad,c='b',linestyle='', marker='x')
ax.plot(ra_rad1,dec_rad1,c='r',alpha=0.5, linestyle='', marker='.')
ax.plot(ra_rad2,dec_rad2,c='g',alpha=0.5, linestyle='', marker='>')
ax.legend(handles=legend_elements, loc='best')
plt.savefig('20190816.png',format='png',dpi=600)
plt.show()


legend_elements = [Line2D([0], [0], marker='.', color='r', label='WithoutPoint-Ast'),Line2D([0], [0], marker='>', color='g', label='Point-Ast')]


fig,(ax1,ax2,ax3)=plt.subplots(1,3, figsize=(16,4))
plt.subplots_adjust(wspace=0.27)
fig.suptitle('Apparent encoder possition with and without pointing model versus astrometric position')
for i in range(len(pos_plot[0])):
    ax1.plot(i,degrees(pos_f_plot[0][i])-degrees(pos_ch_plot[0][i]),'g>')
    ax1.plot(i,degrees(pos_plot[0][i])-degrees(pos_ch_plot[0][i]),'r.')
    # ax1.plot(i,),'bx')

ax1.set_xlabel('Number of point')
ax1.set_ylabel('Hour angle[deg]')
ax1.set_ylim(-0.1,1.6)
ax1.legend(handles=legend_elements, loc='best')

for i in range(len(pos_plot[1])):
    ax2.plot(i,degrees(pos_f_plot[1][i])-degrees(pos_ch_plot[1][i]),'g>')
    ax2.plot(i,degrees(pos_plot[1][i])-degrees(pos_ch_plot[1][i]),'r.')
    # ax2.plot(i,degrees(pos_ch_plot[1][i]),'bx')
ax2.set_xlabel('Number of point')
ax2.set_ylabel('Declination[deg]')
ax2.set_ylim(-0.1,1.6)
ax2.legend(handles=legend_elements, loc='best')
plt.show()



