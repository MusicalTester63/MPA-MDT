import numpy as np
from scipy.optimize import curve_fit,minimize,leastsq
import matplotlib.pyplot as plt
import json
from astropy.coordinates import EarthLocation, SkyCoord
import os

"""
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
"""

true_params = [0.3, -0.2, 0.15, 0.1, 0.05, -0.05, 0.02, -0.01, 0.03]  # Skutočné chyby montáže

def simulate_observation(n_stars):
    np.random.seed(42)
    HA_kat = np.random.uniform(0, 360, n_stars)          # Hodinový uhol
    DEC_kat = np.random.uniform(-90, 90, n_stars)        # Deklinácia
    katalog = np.column_stack((HA_kat, DEC_kat))

    pozorovane = apply_errors(true_params, katalog, n_stars, noise_scale=0.1)

    return katalog, pozorovane

def apply_errors(params, katalog, n_stars, phi=48, noise_scale=0.1):
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params
    HA_kat, DEC_kat = katalog[:, 0], katalog[:, 1]
    
    # Aplikácia systémových chýb
    HA_poz = (
        HA_kat 
        + ZH 
        + CO / np.cos(np.radians(DEC_kat)) 
        + NP * np.tan(np.radians(DEC_kat))
        - MA * np.cos(np.radians(HA_kat)) * np.tan(np.radians(DEC_kat))
        + ME * np.sin(np.radians(HA_kat)) * np.tan(np.radians(DEC_kat))
        + TF * np.cos(np.radians(phi)) * np.sin(np.radians(HA_kat)) / np.cos(np.radians(DEC_kat))
        - DF * (np.cos(np.radians(phi)) * np.cos(np.radians(HA_kat)) + np.sin(np.radians(phi)) * np.tan(np.radians(DEC_kat)))
    )
    
    DEC_poz = (
        DEC_kat 
        + ZD 
        + MA * np.sin(np.radians(HA_kat)) 
        + ME * np.cos(np.radians(HA_kat))
        + TF * (np.cos(np.radians(phi)) * np.cos(np.radians(HA_kat)) * np.sin(np.radians(DEC_kat)) 
        - np.sin(np.radians(phi)) * np.cos(np.radians(DEC_kat)))
        + FO * np.cos(np.radians(HA_kat))
    )
    
    # Pridajte náhodný šum (napr. Gaussovský s σ=0.1°)
    HA_poz += np.random.normal(0, noise_scale, n_stars)
    DEC_poz += np.random.normal(0, noise_scale, n_stars)
    
    return np.column_stack((HA_poz, DEC_poz))

# Výpočet rezíduí pred a po korekcii
def compute_residuals(katalog, pozorovane, ha_corrected, dec_corrected):
    # Rezíduá pred korekciou
    res_ha_before = np.abs(pozorovane[:, 0] - katalog[:, 0])
    res_dec_before = np.abs(pozorovane[:, 1] - katalog[:, 1])
    
    # Rezíduá po korekcii
    res_ha_after = np.abs(ha_corrected - katalog[:, 0])
    res_dec_after = np.abs(dec_corrected - katalog[:, 1])
    
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

iterations = [50, 100, 200, 400, 800, 1600, 3200]

n_stars = 1600

katalog, pozorovane = simulate_observation(n_stars)

print(f"{'HA':>10}\t\t{'DEC':>10}\t\t\t{'HA_o':>10}\t\t\t{'DEC_o':>10}")
for (ha, dec), (ha_o, dec_o) in zip(katalog, pozorovane):
    print(f"{ha:10.4f}\t\t{dec:10.4f}\t\t\t{ha_o:10.4f}\t\t\t{dec_o:10.4f}")

print("\n")

optimal_params, _ = leastsq(error_function, initial_params, args=(katalog, pozorovane))
ha_c, dec_c = calculate_corrected_coords(optimal_params, pozorovane, 48)

# Vypočítajte reziduá
res_before, res_after = compute_residuals(katalog, pozorovane, ha_c, dec_c)

# Rozbaľte reziduá
res_ha_before, res_dec_before = res_before
res_ha_after, res_dec_after = res_after

# Vypíšte hlavičku tabuľky
print(f"{'RES_HA':>10}\t\t{'RES_DEC':>10}\t\t\t{'RES_HA_c':>10}\t\t\t{'RES_DEC_c':>10}")

# Vypíšte hodnoty
for ha, dec, ha_c, dec_c in zip(res_ha_before, res_ha_after, res_ha_after, res_dec_after):
    print(f"{ha:10.4f}\t\t{dec:10.4f}\t\t\t{ha_c:10.4f}\t\t\t{dec_c:10.4f}")

ha_c, dec_c = calculate_corrected_coords(optimal_params, pozorovane, 48)
res_before, res_after = compute_residuals(katalog, pozorovane, ha_c, dec_c)

plot_residuals(res_before, res_after)




# Výpis výsledkov
ZH_c, ZD_c, CO_c, NP_c, MA_c, ME_c, TF_c, DF_c, FO_c = optimal_params
ZH_t, ZD_t, CO_t, NP_t, MA_t, ME_t, TF_t, DF_t, FO_t = true_params

print(f"Optimalizované korekčné faktory montáže pri {n_stars} pozorovaniach:")
print(f"ZH (nulový bod hodinového uhla): δ_ZH={abs(ZH_c - ZH_t):.4f}")
print(f"ZD (nulový bod deklinácie): δ_ZD={abs(ZD_c - ZD_t):.4f}")
print(f"CO (kolimácia): δ_CO={abs(CO_c - CO_t):.4f}")
print(f"NP (nekolmosť osí): δ_NP={abs(NP_c - NP_c):.4f}")
print(f"MA (chyba vyrovnania E-W): δ_MA={abs(MA_c - MA_t):.4f}")
print(f"ME (chyba vyrovnania N-S): δ_ME={abs(ME_c - ME_t):.4f}")
print(f"TF (priehyb tubusu): δ_TF={abs(TF_c - TF_t):.4f}")
print(f"DF (chyba deklinácie): δ_DF={abs(DF_c - DF_t):.4f}")
print(f"FO (chyba vidlice): δ_FO={abs(FO_c - FO_t):.4f}")
print("\n")







"""
for i in iterations:
    katalog, pozorovane = simulate_observation(i)
    # Nájdeme optimálne korekčné faktory
    optimal_params, _ = leastsq(error_function, initial_params, args=(katalog, pozorovane))

    # Výpis výsledkov
    ZH_c, ZD_c, CO_c, NP_c, MA_c, ME_c, TF_c, DF_c, FO_c = optimal_params
    ZH_t, ZD_t, CO_t, NP_t, MA_t, ME_t, TF_t, DF_t, FO_t = true_params

    print(f"Optimalizované korekčné faktory montáže pri {i} pozorovaniach:")
    print(f"ZH (nulový bod hodinového uhla): δ_ZH={abs(ZH_c - ZH_t):.4f}")
    print(f"ZD (nulový bod deklinácie): δ_ZD={abs(ZD_c - ZD_t):.4f}")
    print(f"CO (kolimácia): δ_CO={abs(CO_c - CO_t):.4f}")
    print(f"NP (nekolmosť osí): δ_NP={abs(NP_c - NP_c):.4f}")
    print(f"MA (chyba vyrovnania E-W): δ_MA={abs(MA_c - MA_t):.4f}")
    print(f"ME (chyba vyrovnania N-S): δ_ME={abs(ME_c - ME_t):.4f}")
    print(f"TF (priehyb tubusu): δ_TF={abs(TF_c - TF_t):.4f}")
    print(f"DF (chyba deklinácie): δ_DF={abs(DF_c - DF_t):.4f}")
    print(f"FO (chyba vidlice): δ_FO={abs(FO_c - FO_t):.4f}")
    print("\n")

    ha_c, dec_c = calculate_corrected_coords(optimal_params, pozorovane, 48)
    res_before, res_after = compute_residuals(katalog, pozorovane, ha_c, dec_c)

    plot_residuals(res_before, res_after)
"""






