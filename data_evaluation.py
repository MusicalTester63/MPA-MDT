import numpy as np
from scipy.optimize import curve_fit,minimize,leastsq
import matplotlib.pyplot as plt
import json
from astropy.coordinates import EarthLocation, SkyCoord
import os

# Načítanie dát zo súboru
def load_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)  # Preskočí hlavičku
    katalog = data[:, :2]  # HA a DEC (katalógové súradnice)
    pozorovane = data[:, 2:]  # HA_PNT a DEC_PNT (pozorované súradnice)
    return katalog, pozorovane

# Cesta k súboru
file_path = "data.txt"

# Načítanie dát
katalog, pozorovane = load_data(file_path)



# Výpočet rezíduí pred a po korekcii
def compute_residuals(katalog, pozorovane, ha_corrected, dec_corrected):
    # Rezíduá pred korekciou
    res_ha_before = pozorovane[:, 0] - katalog[:, 0]
    res_dec_before = pozorovane[:, 1] - katalog[:, 1]
    
    # Rezíduá po korekcii
    res_ha_after = ha_corrected - katalog[:, 0]
    res_dec_after = dec_corrected - katalog[:, 1]
    
    return (res_ha_before, res_dec_before), (res_ha_after, res_dec_after)






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


def error_function_lm(params, katalog, pozorovane):
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params
    t_kat, d_kat = katalog[:, 0], katalog[:, 1]
    t_poz, d_poz = pozorovane[:, 0], pozorovane[:, 1]
    
    # Korekcia pre hodinový uhol t a deklináciu δ
    t_corr = t_poz - (t_kat + ZH + CO / np.cos(np.radians(d_kat)) + NP * np.tan(np.radians(d_kat))
                     - MA * np.cos(np.radians(t_kat)) * np.tan(np.radians(d_kat))
                     + ME * np.sin(np.radians(t_kat)) * np.tan(np.radians(d_kat))
                     + TF * np.cos(np.radians(48)) * np.sin(np.radians(t_kat)) / np.cos(np.radians(d_kat))
                     - DF * (np.cos(np.radians(48)) * np.cos(np.radians(t_kat)) + np.sin(np.radians(48)) * np.tan(np.radians(d_kat)) ))
    
    d_corr = d_poz - (d_kat + ZD + MA * np.sin(np.radians(t_kat)) + ME * np.cos(np.radians(t_kat))
                     + TF * (np.cos(np.radians(48)) * np.cos(np.radians(t_kat)) * np.sin(np.radians(d_kat)) 
                     - np.sin(np.radians(48)) * np.cos(np.radians(d_kat)))
                     + FO * np.cos(np.radians(t_kat)))
    
    return np.concatenate((t_corr, d_corr))


def plot_residuals(res_before, res_after):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Hodinový uhol (HA)
    ax[0].scatter(range(len(res_before[0])), res_before[0], c='r', label='Pred korekciou')
    ax[0].scatter(range(len(res_after[0])), res_after[0], c='g', marker='x', label='Po korekcii')
    ax[0].set_title('Rezíduá hodinového uhla (HA)')
    ax[0].set_ylabel('Odchýlka [°]')
    ax[0].legend()
    ax[0].grid(True)
    
    # Deklinácia (DEC)
    ax[1].scatter(range(len(res_before[1])), res_before[1], c='r', label='Pred korekciou')
    ax[1].scatter(range(len(res_after[1])), res_after[1], c='g', marker='x', label='Po korekcii')
    ax[1].set_title('Rezíduá deklinácie (DEC)')
    ax[1].set_xlabel('Číslo hviezdy')
    ax[1].set_ylabel('Odchýlka [°]')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Počiatočné odhady korekčných faktorov (9 neznámych parametrov)
initial_params = np.zeros(9)


def objective(params, katalog, pozorovane):
    # error_function_lm vráti pole rezíduí (pre HA aj DEC)
    residuals = error_function_lm(params, katalog, pozorovane)
    # Súčet štvorcov rezíduí (vlastne, chi-kvadrát)
    return np.sum(residuals**2)

# Nájdeme optimálne korekčné faktory
#result  = minimize(objective, initial_params, args=(katalog, pozorovane), method='L-BFGS-B')
#optimal_params = result.x
optimal_params, _ = leastsq(error_function, initial_params, args=(katalog, pozorovane))
#optimal_params, _ = curve_fit(lambda x, *params: error_function_lm(params, x[0], x[1]), (katalog, pozorovane), np.zeros(len(katalog)*2), p0=initial_params)

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

res_before, res_after = compute_residuals(katalog, pozorovane, ha_corrected, dec_corrected)
plot_residuals(res_before, res_after)


def sum_squared_error(params, katalog, pozorovane):
    residuals = error_function_lm(params, katalog, pozorovane)
    return np.sum(residuals**2)
ZH_opt = optimal_params[0]
ZH_vals = np.linspace(ZH_opt - 0.1, ZH_opt + 0.1, 100)  # upravte interval podľa potreby

errors = []
for z in ZH_vals:
    params = optimal_params.copy()
    params[0] = z  # variujeme len ZH
    errors.append(sum_squared_error(params, katalog, pozorovane))

plt.figure(figsize=(8, 5))
plt.plot(ZH_vals, errors, 'b-', label='Suma štvorcov rezíduí')
plt.xlabel("ZH (nulový bod hodinového uhla)")
plt.ylabel("Suma štvorcov rezíduí")
plt.title("Závislosť error funkcie na ZH")
plt.legend()
plt.grid(True)
plt.show()