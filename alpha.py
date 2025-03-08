import numpy as np
from scipy.optimize import leastsq

# Ukážkové dáta: Katalógové a pozorované súradnice hviezd
# (hodinový uhol t v stupňoch, deklinácia δ v stupňoch)
katalog = np.array([
    [10, 30], 
    [20, 40], 
    [30, 50], 
    [40, 35], 
    [50, 45],
    [60, 60], 
    [70, 65], 
    [80, 55], 
    [90, 25], 
    [100, 20],
    [110, 30], 
    [120, 40], 
    [130, 50], 
    [140, 35], 
    [150, 45],
    [160, 60], 
    [170, 65], 
    [180, 55], 
    [190, 25], 
    [200, 20],
])

pozorovane = np.array([
    [10.5, 30.3],
    [20.8, 40.5], 
    [29.5, 49.7], 
    [39.7, 34.9], 
    [50.2, 44.8],
    [60.3, 60.5], 
    [70.2, 65.1], 
    [80.1, 54.9], 
    [89.9, 24.8], 
    [99.8, 20.4],
    [109.7, 30.2], 
    [119.6, 40.3], 
    [129.9, 49.8], 
    [139.7, 35.1], 
    [149.4, 45.3],
    [159.5, 59.9], 
    [169.8, 64.7], 
    [179.6, 54.3], 
    [189.5, 25.0], 
    [199.2, 20.1],
])

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
                     - DF * (np.cos(np.radians(48)) * np.cos(np.radians(t_kat)) + np.sin(np.radians(48)) * np.tan(np.radians(d_kat)) ))
    
    d_corr = d_poz - (d_kat + ZD + MA * np.sin(np.radians(t_kat)) + ME * np.cos(np.radians(t_kat))
                     + TF * (np.cos(np.radians(48)) * np.cos(np.radians(t_kat)) * np.sin(np.radians(d_kat)) - np.sin(np.radians(48)) * np.cos(np.radians(d_kat)))
                     + FO * np.cos(np.radians(t_kat)))
    
    return np.concatenate((t_corr, d_corr))


def calculate_corrected_coords(params, pozorovane, phi):
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params
    t_poz, d_poz = pozorovane[:, 0], pozorovane[:, 1]
    
    # Korekcia: Od nameraných hodnôt ODČÍTAME chyby montáže
    t_corr = t_poz - (ZH + CO / np.cos(np.radians(d_poz)) + NP * np.tan(np.radians(d_poz))
                     - MA * np.cos(np.radians(t_poz)) * np.tan(np.radians(d_poz))
                     + ME * np.sin(np.radians(t_poz)) * np.tan(np.radians(d_poz))
                     + TF * np.cos(np.radians(phi)) * np.sin(np.radians(t_poz)) / np.cos(np.radians(d_poz))
                     - DF * (np.cos(np.radians(phi)) * np.cos(np.radians(t_poz)) + np.sin(np.radians(phi)) * np.tan(np.radians(d_poz)) )
    )
    
    d_corr = d_poz - (ZD + MA * np.sin(np.radians(t_poz)) + ME * np.cos(np.radians(t_poz))
                     + TF * (np.cos(np.radians(phi)) * np.cos(np.radians(t_poz)) * np.sin(np.radians(d_poz)) 
                     - np.sin(np.radians(phi)) * np.cos(np.radians(d_poz)))
                     + FO * np.cos(np.radians(t_poz)))
    
    return t_corr, d_corr

# Počiatočné odhady korekčných faktorov (9 neznámych parametrov)
initial_params = np.zeros(9)

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

# Výpočet opravených hodnôt HA a DEC
ha_corrected, dec_corrected = calculate_corrected_coords(optimal_params, pozorovane, 48)

# Výpis porovnania údajov
print("katalog HA DEC, HA_namerane DEC_namerane, HA_opravene DEC_opravene")
for i in range(len(katalog)):
    print(f"{katalog[i][0]:.6f} {katalog[i][1]:.6f}, {pozorovane[i][0]:.6f} {pozorovane[i][1]:.6f}, {ha_corrected[i]:.6f} {dec_corrected[i]:.6f}")
