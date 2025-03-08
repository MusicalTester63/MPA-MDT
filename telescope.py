import numpy as np
from scipy.optimize import leastsq

# Načítanie dát zo súboru
def load_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)  # Preskočí hlavičku
    katalog = data[:, :2]  # HA a DEC (katalógové súradnice)
    pozorovane = data[:, 2:]  # HA_PNT a DEC_PNT (pozorované súradnice)
    return katalog, pozorovane

# Cesta k súboru
file_path = "point_data1.txt"

# Načítanie dát
katalog, pozorovane = load_data(file_path)

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