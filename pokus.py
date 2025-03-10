import numpy as np
import matplotlib.pyplot as plt

# Simulácia dátových súborov
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
    
    # Pridajte náhodný šum
    HA_poz += np.random.normal(0, noise_scale, n_stars)
    DEC_poz += np.random.normal(0, noise_scale, n_stars)
    
    return np.column_stack((HA_poz, DEC_poz))

# Vytvorenie modelovej matice X a vektora Y
def create_design_matrix(katalog, phi=48):
    HA_kat = np.radians(katalog[:, 0])
    DEC_kat = np.radians(katalog[:, 1])
    n = len(HA_kat)
    
    X = np.zeros((2*n, 9))  # 2 rovnice na pozorovanie
    
    for i in range(n):
        # Rovnica pre HA (t_e - t)
        X[2*i, 0] = 1  # ZH
        X[2*i, 2] = 1 / np.cos(DEC_kat[i])  # CO
        X[2*i, 3] = np.tan(DEC_kat[i])  # NP
        X[2*i, 4] = -np.cos(HA_kat[i]) * np.tan(DEC_kat[i])  # MA
        X[2*i, 5] = np.sin(HA_kat[i]) * np.tan(DEC_kat[i])  # ME
        X[2*i, 6] = np.cos(np.radians(phi)) * np.sin(HA_kat[i]) / np.cos(DEC_kat[i])  # TF
        X[2*i, 7] = -(np.cos(np.radians(phi)) * np.cos(HA_kat[i]) + np.sin(np.radians(phi)) * np.tan(DEC_kat[i]))  # DF
        
        # Rovnica pre DEC (delta_e - delta)
        X[2*i+1, 1] = 1  # ZD
        X[2*i+1, 4] = np.sin(HA_kat[i])  # MA
        X[2*i+1, 5] = np.cos(HA_kat[i])  # ME
        X[2*i+1, 6] = (np.cos(np.radians(phi)) * np.cos(HA_kat[i]) * np.sin(DEC_kat[i]) 
                        - np.sin(np.radians(phi)) * np.cos(DEC_kat[i]))  # TF
        X[2*i+1, 8] = np.cos(HA_kat[i])  # FO
    
    return X

# Hlavný skript
n_stars = 1000
katalog, pozorovane = simulate_observation(n_stars)
Y = (pozorovane - katalog).flatten()  # Rezíduá (pozorovane - katalog)

X = create_design_matrix(katalog)
sigma = 0.1  # Šum z simulácie
weights = 1/(sigma**2) * np.ones_like(Y)  # Váhy pre najmenšie štvorce

# Riešenie pomocou váženej LLSQ
ATA = X.T @ (weights[:, None] * X)
ATY = X.T @ (weights * Y)
params_estimated = np.linalg.solve(ATA, ATY)

# Výpočet rezíduí po korekcii
corrections = X @ params_estimated
residuals_corrected = Y - corrections

# Výpis výsledkov
param_names = ['ZH', 'ZD', 'CO', 'NP', 'MA', 'ME', 'TF', 'DF', 'FO']
print("Parameter\tOdhadnutá hodnota\tSkutočná hodnota\tOdchýlka")
for name, est, true in zip(param_names, params_estimated, true_params):
    print(f"{name}\t\t{est:.4f}\t\t\t{true:.4f}\t\t\t{abs(est - true):.4f}")

# Vizualizácia rezíduí
plt.figure(figsize=(12, 6))





# Pred korekciou
plt.subplot(1, 2, 1)
ha_res_before = Y[::2]    # Každý druhý prvok od 0: HA reziduá
dec_res_before = Y[1::2]  # Každý druhý prvok od 1: DEC reziduá
plt.scatter(ha_res_before, dec_res_before, alpha=0.5, s=10)
plt.axhline(0, color='k', lw=0.5)  # Horizontálna čiara na y=0
plt.axvline(0, color='k', lw=0.5)  # Vertikálna čiara na x=0
plt.xlabel('HA reziduum [°]')
plt.ylabel('DEC reziduum [°]')
plt.title('Pred korekciou')
max_val = np.max(np.abs([ha_res_before, dec_res_before])) * 1.1
plt.xlim(-max_val, max_val)
plt.ylim(-max_val, max_val)
plt.grid(True, linestyle='--', alpha=0.3)

# Po korekcii
plt.subplot(1, 2, 2)
ha_res_after = residuals_corrected[::2]    
dec_res_after = residuals_corrected[1::2]  
plt.scatter(ha_res_after, dec_res_after, alpha=0.5, s=10, color='orange')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.xlabel('HA reziduum [°]')
plt.ylabel('DEC reziduum [°]')
plt.title('Po korekcii')
max_val = np.max(np.abs([ha_res_after, dec_res_after])) * 1.1
plt.xlim(-max_val, max_val)
plt.ylim(-max_val, max_val)
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()