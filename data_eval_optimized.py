import json
import os
import numpy as np
from scipy.optimize import minimize
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
import matplotlib.pyplot as plt
from point_eqs import *
from matplotlib.lines import Line2D

# Nastavenie globálnych parametrov
plt.rcParams.update({'font.size': 14})

# Konštanty
EARTH_LOCATION = EarthLocation(lat=48.3733*u.deg, lon=17.240*u.deg, height=531.1*u.m)
PHI = EARTH_LOCATION.lat.rad

def load_data(mode):
    """Načíta dáta z textového súboru a vráti merané a astrometrické pozície."""
    with open('data.txt', 'r') as f:
        lines = f.readlines()[1:]
    
    if mode == '1':
        meas_pos = [[], []]
        pos_ast = [[], []]
        for line in lines:
            if line.strip():
                ra_ast, dec_ast, ra_meas, dec_meas = map(float, line.split('\t'))
                meas_pos[0].append(np.radians(ra_meas))
                meas_pos[1].append(np.radians(dec_meas))
                pos_ast[0].append(np.radians(ra_ast))
                pos_ast[1].append(np.radians(dec_ast))
    elif mode == '2':
        meas_pos = []
        pos_ast = []
        for line in lines:
            ra_ast, dec_ast, ra_meas, dec_meas = map(float, line.split('\t'))
            meas_pos.append(np.radians(np.array([ra_meas, dec_meas])))
            pos_ast.append(np.radians(np.array([ra_ast, dec_ast])))
    
    return np.array(meas_pos), np.array(pos_ast)

def calculate_point(t, dec, zh, zd, ma, me, mp, tf, co, df, fo):
    """Vypočíta opravené pozície na základe modelu."""
    tc = t + zh + co/np.cos(dec) + mp*np.tan(dec) - ma*np.cos(t)*np.tan(dec) + me*np.sin(t)*np.tan(dec) + tf*np.cos(PHI)*np.sin(t)/np.cos(dec) - df*(np.cos(PHI)*np.cos(t) + np.sin(PHI)*np.tan(dec))
    decc = dec + zd + ma*np.sin(t) + me*np.cos(t) + tf*(np.cos(PHI)*np.cos(t)*np.sin(dec) - np.sin(PHI)*np.cos(dec)) + fo*np.cos(t)
    return np.array([tc, decc])

def rms(meas_pos, pos_ast):
    """Vypočíta RMS (Root Mean Square) medzi dvoma polohami."""
    return np.sqrt(np.mean((meas_pos - pos_ast) ** 2))

def metric(par, *args):
    """Metrika pre optimalizáciu."""
    data = list(load_data('1'))
    
    if args:
        data[0] = data[0][0:2, args[0]:args[1]:1]
        data[1] = data[1][0:2, args[0]:args[1]:1]
    
    calcul = calculate_point(t=data[0][0], dec=data[0][1], zh=par[0], zd=par[1], ma=par[2], me=par[3], mp=par[4], tf=par[5], co=par[6], df=par[7], fo=par[8])
    scalar = np.sum(((calcul[0] - data[1][0])/0.0001)**2 + ((calcul[1] - data[1][1])/0.0001)**2)
    return scalar

def plot_results(pos_plot, pos_ch_plot, pos_f_plot, title, filename):
    """Vykreslí výsledky."""
    legend_elements = [
        Line2D([0], [0], marker='>', color='g', label=f'Point-Ast\nRMS={np.round(rms(pos_f_plot, pos_ch_plot)*3600, 3)}"'),
        Line2D([0], [0], marker='o', color='w', label='0.5deg', markeredgecolor='r', markerfacecolor='w', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='0.25deg', markeredgecolor='g', markerfacecolor='w', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='0.05deg', markeredgecolor='b', markerfacecolor='w', markersize=15)
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title)
    
    for ax in (ax1, ax2):
        ax.axhline(y=0, color="black")
        ax.axvline(x=0, color="black")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.add_patch(plt.Circle((0, 0), 0.5, color='r', fill=False, label='0.5deg'))
        ax.add_patch(plt.Circle((0, 0), 0.25, color='g', fill=False, label='0.25deg'))
        ax.add_patch(plt.Circle((0, 0), 0.05, color='b', fill=False, label='0.05deg'))
    
    for i in range(len(pos_plot[0])):
        ax1.plot(np.degrees(pos_plot[0][i]) - np.degrees(pos_ch_plot[0][i]), np.degrees(pos_plot[1][i]) - np.degrees(pos_ch_plot[1][i]), 'r.')
        ax2.plot(np.degrees(pos_f_plot[0][i]) - np.degrees(pos_ch_plot[0][i]), np.degrees(pos_f_plot[1][i]) - np.degrees(pos_ch_plot[1][i]), 'g>')
    
    ax1.set_xlabel('Pre-pointing hour angle distance [deg]')
    ax1.set_ylabel('Pre-pointing declination distance [deg]')
    ax1.legend(handles=legend_elements, loc='best')
    
    ax2.set_xlabel('Post-pointing hour angle distance [deg]')
    ax2.set_ylabel('Post-pointing declination distance [deg]')
    ax2.legend(handles=legend_elements, loc='best')
    
    plt.savefig(filename, format='png', dpi=600)
    plt.show()

def save_params(params):
    """Uloží parametre do JSON súboru."""
    names = ['zh', 'zd', 'ma', 'me', 'mp', 'tf', 'co', 'df', 'fo']
    with open('AGO70_params.json', 'w') as js:
        json.dump({key: params[i] for i, key in enumerate(names)}, js, indent=3)
    print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
    print('Params was saved -> AGO70_params.json')
    print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+')
    print('Name \t Value')
    for i, key in enumerate(names):
        print(key, '\t', np.round(np.degrees(params[i])*3600, 3), 'arc-sec')
    print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+')

def main():
    meas_pos, pos_ast = load_data('1')
    n_points = meas_pos.shape[1]
    steps = [(i, min(i + 15, n_points)) for i in range(0, n_points, 15)]
    
    params = np.zeros(9)
    for k, (start, end) in enumerate(steps):
        result = minimize(metric, params, args=(start, end), method='nelder-mead', options={'xatol': 1e-7, 'maxiter': 10000, 'disp': False})
        params = result.x
        pos_f = calculate_point(meas_pos[0], meas_pos[1], *params)
        
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        print('RMS-without PM:', rms(meas_pos, pos_ast)*3600, '"')
        print(f'After {k} iterations, accounting {end} points')
        print('Current RMS-with PM:', rms(pos_f, pos_ast)*3600, '"')
        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
        
        plot_results(meas_pos, pos_ast, pos_f, f'Apparent encoder position with and without pointing model versus astrometric position - N = {end}', f'PM-iter-{k}-part.png')
    
    save_params(params)
    
    # Ďalšie vykreslenie výsledkov
    c = SkyCoord(ra=meas_pos[0], dec=meas_pos[1], unit=(u.rad, u.rad))
    c1 = SkyCoord(ra=pos_ast[0], dec=pos_ast[1], unit=(u.rad, u.rad))
    c2 = SkyCoord(ra=pos_f[0], dec=pos_f[1], unit=(u.rad, u.rad))
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_rlim(1.6, -0.5, 0.2)
    ax.plot(c.ra.radian, c.dec.radian, 'bx', label='Ast, pos.')
    ax.plot(c1.ra.radian, c1.dec.radian, 'r.', alpha=0.5, label='Without point.')
    ax.plot(c2.ra.radian, c2.dec.radian, 'g>', alpha=0.5, label='With point')
    ax.legend(loc='best')
    plt.savefig('20190816.png', format='png', dpi=600)
    plt.show()

if __name__ == "__main__":
    main()