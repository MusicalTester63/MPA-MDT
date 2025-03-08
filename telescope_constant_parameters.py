import numpy as np
from scipy.optimize import leastsq

# Ukážkové dáta: Katalógové a pozorované súradnice hviezd
# (hodinový uhol t v stupňoch, deklinácia δ v stupňoch)
katalog = np.array([
    [359.8530427327059,     49.36184149234271],
    [359.05954480923975,    49.36207248006128],
    [358.6467350637908,     48.57567696971334],
    [358.74256073908873,    47.99522974838695],
    [359.3017866600755,     47.54428301345832],
    [0.1277384323365709,     47.37807684850538],
    [0.978417022132362,      47.54590200873345],
    [1.513008439633495,      47.99496355038474],
    [1.6014787984936447,     48.57588640434903],
    [1.1810422456073297,     49.09160925256674]
])

pozorovane = np.array([
    [359.8519176967054,     49.361391423359585],
    [359.0575030825711,     49.36128902051017],
    [358.64023462355163,    48.575198809959886],
    [358.7329768954252,     47.99364028693149],
    [359.29074415598086,    47.54451015360522],
    [0.11848794709345611,   47.376598520177566],
    [0.9680831196005784,    47.54449028274771],
    [1.5038412941393062,    47.99356855392464],
    [1.5964368918475316,    48.57501942885753],
    [1.1790005344985275,    49.090936901376075]
])

# Počiatočné odhady korekčných faktorov (9 neznámych parametrov)
initial_params = np.zeros(9)

# Modelová funkcia pre chyby montáže
def error_function(params, katalog, pozorovane):
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params
    t_kat, d_kat = katalog[:, 0], katalog[:, 1]
    t_poz, d_poz = pozorovane[:, 0], pozorovane[:, 1]
    
    # Oprava hodinového uhla t a deklinácie δ
    t_corr = t_poz - (t_kat + ZH + CO / np.cos(np.radians(d_kat)) + NP * np.tan(np.radians(d_kat))
                     - MA * np.cos(np.radians(t_kat)) * np.tan(np.radians(d_kat))
                     + ME * np.sin(np.radians(t_kat)) * np.tan(np.radians(d_kat))
                     + TF * np.cos(np.radians(48)) * np.sin(np.radians(t_kat)) / np.cos(np.radians(d_kat))
                     - DF * (np.cos(np.radians(48)) * np.cos(np.radians(t_kat)) + np.sin(np.radians(48)) * np.tan(np.radians(d_kat))))
    
    d_corr = d_poz - (d_kat + ZD + MA * np.sin(np.radians(t_kat)) + ME * np.cos(np.radians(t_kat))
                     + TF * (np.cos(np.radians(48)) * np.cos(np.radians(t_kat)) * np.sin(np.radians(d_kat)) - np.sin(np.radians(48)) * np.cos(np.radians(d_kat)))
                     + FO * np.cos(np.radians(t_kat)))
    
    return np.concatenate((t_corr, d_corr))

# Nájdeme optimálne korekčné faktory
optimal_params, _ = leastsq(error_function, initial_params, args=(katalog, pozorovane))

# Výpis výsledkov
ZH, ZD, CO, NP, MA, ME, TF, DF, FO = optimal_params
print("Optimalizované korekčné faktory montáže:")
print(f"ZH (nulový bod hodinového uhla): {ZH:.4f}")
print(f"ZD (nulový bod deklinácie): {ZD:.4f}")
print(f"CO (kolimácia): {CO:.4f}")
print(f"NP (nekolmosť osí): {NP:.4f}")
print(f"MA (chyba vyrovnania E-W): {MA:.4f}")
print(f"ME (chyba vyrovnania N-S): {ME:.4f}")
print(f"TF (priehyb tubusu): {TF:.4f}")
print(f"DF (chyba deklinácie): {DF:.4f}")
print(f"FO (chyba vidlice): {FO:.4f}")
