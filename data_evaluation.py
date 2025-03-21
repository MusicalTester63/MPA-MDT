import numpy as np
from scipy.optimize import curve_fit,minimize,leastsq
import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from geopy.geocoders import Nominatim
import os

true_params = [0.3, -0.2, 0.15, 0.1, 0.05, -0.05, 0.02, -0.01, 0.03]  # Skutočné chyby montáže

def get_latitude():
    try:
        geolocator = Nominatim(user_agent="geo_locator")
        location = geolocator.geocode("Bratislava")
        if location:
            return location.latitude
        else:
            return 48.0
    except Exception as e:
        print(f"Chyba pri geolokácii: {e}")
        return 48.0

def load_data(mode, n_stars):
    """
    Načíta dáta buď zo súboru, alebo vygeneruje simulované pozorovania.

    Parametre:
        mode (str): "file" pre načítanie zo súboru, "simulate" pre generovanie dát.
        n_stars (int): Počet hviezd na simuláciu alebo počet riadkov zo súboru.
    
    Výstup:
        tuple: (katalógové súradnice, pozorované súradnice)
    """
    if n_stars <= 0:
        raise ValueError("Number of observations has to be a non zero value.")

    if mode == "simulate":
        katalog, pozorovane = simulate_observation(n_stars)
        

    elif mode == "file":
        data = np.loadtxt("./data/data.txt", skiprows=1)  # Preskočí hlavičku
        if len(data) < n_stars:
            raise ValueError(f"File contains only {len(data)} rows of data. You required {n_stars}.")
        katalog = data[:n_stars, :2]  # HA a DEC (katalógové súradnice)
        pozorovane = data[:n_stars, 2:]  # HA_PNT a DEC_PNT (pozorované súradnice)

    else:
        raise ValueError('Invalid mode. Please use ".')

    return katalog, pozorovane

def simulate_observation(n_stars):
    # Hodinový uhol (preferovane okolo ± 4 hodín, teda ~±60°)
    HA_kat = np.random.normal(loc=0, scale=60, size=n_stars)
    HA_kat = np.clip(HA_kat, -180, 180)  # Orezanie na platný rozsah

    # Deklinácia (sinusové rozdelenie)
    DEC_kat = np.arcsin(np.random.uniform(-1, 1, n_stars)) * (180 / np.pi)

    katalog = np.column_stack((HA_kat, DEC_kat))

    # Aplikovanie chýb na simulované dáta
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

def plot_polar_comparison(katalog, pozorovane, pozorovane_corrected):
    """
    Vykreslí polárny graf pre katalog, pozorovane a pozorovane_corrected.

    Parametre:
    ----------
    katalog : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty z katalógu.
    pozorovane : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní pred korekciou.
    pozorovane_corrected : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní po korekcii.
    """
    ha_katalog, dec_katalog = katalog[:, 0], katalog[:, 1]
    ha_pozorovane, dec_pozorovane = pozorovane[:, 0], pozorovane[:, 1]
    ha_corrected, dec_corrected = pozorovane_corrected[:, 0], pozorovane_corrected[:, 1]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    ax.scatter(np.radians(ha_katalog), dec_katalog, color='blue', label='Katalóg', alpha=0.6)

    ax.scatter(np.radians(ha_pozorovane), dec_pozorovane, color='red', label='Pozorované (pred korekciou)', alpha=0.6)

    ax.scatter(np.radians(ha_corrected), dec_corrected, color='green', label='Pozorované (po korekcii)', alpha=0.6)

    # Nastavenie grafu
    ax.set_title("Porovnanie pozorovaní pred a po korekcii", pad=20)
    ax.set_theta_zero_location('N')  # Nastavenie severu na vrchol grafu
    ax.set_theta_direction(-1)  # Hodinový uhol sa zväčšuje v smere hodinových ručičiek
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Zobrazenie grafu
    plt.show()

    """
    Vykreslí karteziánsky graf pre katalog, pozorovane a pozorovane_corrected.

    Parametre:
    ----------
    katalog : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty z katalógu.
    pozorovane : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní pred korekciou.
    pozorovane_corrected : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní po korekcii.
    """
    # Rozbaľte dáta
    ha_katalog, dec_katalog = katalog[:, 0], katalog[:, 1]
    ha_pozorovane, dec_pozorovane = pozorovane[:, 0], pozorovane[:, 1]
    ha_corrected, dec_corrected = pozorovane_corrected[:, 0], pozorovane_corrected[:, 1]

    # Vytvorte graf
    plt.figure(figsize=(10, 6))

    # Vykreslite katalógové hodnoty (referenčné)
    plt.scatter(ha_katalog, dec_katalog, color='blue', label='Katalóg', alpha=0.6)

    # Vykreslite pozorované hodnoty pred korekciou
    plt.scatter(ha_pozorovane, dec_pozorovane, color='red', label='Pozorované (pred korekciou)', alpha=0.6)

    # Vykreslite pozorované hodnoty po korekcii
    plt.scatter(ha_corrected, dec_corrected, color='green', label='Pozorované (po korekcii)', alpha=0.6)

    # Nastavenie grafu
    plt.title("Porovnanie pozorovaní pred a po korekcii")
    plt.xlabel("Hodinový uhol (HA) [°]")
    plt.ylabel("Deklinácia (DEC) [°]")
    plt.legend(loc='upper right')
    plt.grid(True)

    # Zobrazenie grafu
    plt.show()

    """
    Vykreslí karteziánsky graf pre katalog, pozorovane a pozorovane_corrected,
    s čiarami spájajúcimi katalógové body s pozorovanými a opravenými bodmi,
    a zobrazí dĺžky týchto úsečiek.

    Parametre:
    ----------
    katalog : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty z katalógu.
    pozorovane : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní pred korekciou.
    pozorovane_corrected : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní po korekcii.
    """
    # Rozbaľte dáta
    ha_katalog, dec_katalog = katalog[:, 0], katalog[:, 1]
    ha_pozorovane, dec_pozorovane = pozorovane[:, 0], pozorovane[:, 1]
    ha_corrected, dec_corrected = pozorovane_corrected[:, 0], pozorovane_corrected[:, 1]

    # Vytvorte graf
    plt.figure(figsize=(10, 6))

    # Vykreslite katalógové hodnoty (referenčné)
    plt.scatter(ha_katalog, dec_katalog, color='blue', label='Katalóg', alpha=0.6)

    # Vykreslite pozorované hodnoty pred korekciou a spojte ich čiarami s katalógovými bodmi
    plt.scatter(ha_pozorovane, dec_pozorovane, color='red', label='Pozorované (pred korekciou)', alpha=0.6)
    for ha_k, dec_k, ha_p, dec_p in zip(ha_katalog, dec_katalog, ha_pozorovane, dec_pozorovane):
        plt.plot([ha_k, ha_p], [dec_k, dec_p], color='red', linestyle='--', linewidth=0.5, alpha=0.4)
        # Vypočítajte vzdialenosť
        distance = np.sqrt((ha_p - ha_k)**2 + (dec_p - dec_k)**2)
        # Zobrazte vzdialenosť ako text v strede úsečky
        plt.text((ha_k + ha_p) / 2, (dec_k + dec_p) / 2, f"{distance:.2f}", color='red', fontsize=8)

    # Vykreslite pozorované hodnoty po korekcii a spojte ich čiarami s katalógovými bodmi
    plt.scatter(ha_corrected, dec_corrected, color='green', label='Pozorované (po korekcii)', alpha=0.6)
    for ha_k, dec_k, ha_c, dec_c in zip(ha_katalog, dec_katalog, ha_corrected, dec_corrected):
        plt.plot([ha_k, ha_c], [dec_k, dec_c], color='green', linestyle='--', linewidth=0.5, alpha=0.4)
        # Vypočítajte vzdialenosť
        distance = np.sqrt((ha_c - ha_k)**2 + (dec_c - dec_k)**2)
        # Zobrazte vzdialenosť ako text v strede úsečky
        plt.text((ha_k + ha_c) / 2, (dec_k + dec_c) / 2, f"{distance:.2f}", color='green', fontsize=8)

    # Nastavenie grafu
    plt.title("Porovnanie pozorovaní pred a po korekcii s vyznačenými vzdialenosťami")
    plt.xlabel("Hodinový uhol (HA) [°]")
    plt.ylabel("Deklinácia (DEC) [°]")
    plt.legend(loc='upper right')
    plt.grid(True)

    # Zobrazenie grafu
    plt.show()

def plot_cartesian_comparison_with_error_components(katalog, pozorovane, pozorovane_corrected):
    """
    Vykreslí karteziánsky graf pre katalog, pozorovane a pozorovane_corrected,
    s čiarami znázorňujúcimi chyby v HA a DEC.

    Parametre:
    ----------
    katalog : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty z katalógu.
    pozorovane : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní pred korekciou.
    pozorovane_corrected : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní po korekcii.
    """
    ha_katalog, dec_katalog = katalog[:, 0], katalog[:, 1]
    ha_pozorovane, dec_pozorovane = pozorovane[:, 0], pozorovane[:, 1]
    ha_corrected, dec_corrected = pozorovane_corrected[:, 0], pozorovane_corrected[:, 1]

    plt.figure(figsize=(12, 8))

    plt.scatter(ha_katalog, dec_katalog, color='blue', label='Katalóg', alpha=0.6)

    plt.scatter(ha_pozorovane, dec_pozorovane, color='red', label='Pozorované (pred korekciou)', alpha=0.6)
    for ha_k, dec_k, ha_p, dec_p in zip(ha_katalog, dec_katalog, ha_pozorovane, dec_pozorovane):
        
        plt.plot([ha_k, ha_p], [dec_k, dec_k], color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Chyba HA' if ha_k == ha_katalog[0] else "")
        
        plt.plot([ha_p, ha_p], [dec_k, dec_p], color='purple', linestyle='--', linewidth=1, alpha=0.6, label='Chyba DEC' if ha_k == ha_katalog[0] else "")
        
        plt.text(ha_p, dec_k, f"HA: {ha_p - ha_k:.2f}", color='orange', fontsize=8, ha='left', va='bottom')
        plt.text(ha_p, dec_p, f"DEC: {dec_p - dec_k:.2f}", color='purple', fontsize=8, ha='left', va='bottom')

    plt.scatter(ha_corrected, dec_corrected, color='green', label='Pozorované (po korekcii)', alpha=0.6)
    for ha_k, dec_k, ha_c, dec_c in zip(ha_katalog, dec_katalog, ha_corrected, dec_corrected):
        
        plt.plot([ha_k, ha_c], [dec_k, dec_k], color='orange', linestyle='--', linewidth=1, alpha=0.6)
        
        plt.plot([ha_c, ha_c], [dec_k, dec_c], color='purple', linestyle='--', linewidth=1, alpha=0.6)
        
        plt.text(ha_c, dec_k, f"HA: {ha_c - ha_k:.2f}", color='orange', fontsize=8, ha='left', va='bottom')
        plt.text(ha_c, dec_c, f"DEC: {dec_c - dec_k:.2f}", color='purple', fontsize=8, ha='left', va='bottom')

    plt.title("Porovnanie pozorovaní pred a po korekcii s chybami v HA a DEC")
    plt.xlabel("Hodinový uhol (HA) [°]")
    plt.ylabel("Deklinácia (DEC) [°]")
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.show()

def compute_residuals(katalog, pozorovane, pozorovane_corrected):
    # Rezíduá pred korekciou
    res_ha_before = pozorovane[:, 0] - katalog[:, 0]
    res_dec_before = pozorovane[:, 1] - katalog[:, 1]
    
    # Rezíduá po korekcii
    res_ha_after = pozorovane_corrected[:, 0] - katalog[:, 0]
    res_dec_after = pozorovane_corrected[:, 1] - katalog[:, 1]
    
    return (res_ha_before, res_dec_before), (res_ha_after, res_dec_after)

def error_function(params, katalog, pozorovane, latitude):
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params

    t_kat, d_kat = katalog[:, 0], katalog[:, 1]
    t_poz, d_poz = pozorovane[:, 0], pozorovane[:, 1]
    
    t_corr = t_poz - (t_kat + ZH + CO / np.cos(np.radians(d_kat)) + NP * np.tan(np.radians(d_kat))
                     - MA * np.cos(np.radians(t_kat)) * np.tan(np.radians(d_kat))
                     + ME * np.sin(np.radians(t_kat)) * np.tan(np.radians(d_kat))
                     + TF * np.cos(np.radians(latitude)) * np.sin(np.radians(t_kat)) / np.cos(np.radians(d_kat))
                     - DF * (np.cos(np.radians(latitude)) * np.cos(np.radians(t_kat)) + np.sin(np.radians(latitude)) * np.tan(np.radians(d_kat)) ))
    
    d_corr = d_poz - (d_kat + ZD + MA * np.sin(np.radians(t_kat)) + ME * np.cos(np.radians(t_kat))
                     + TF * (np.cos(np.radians(latitude)) * np.cos(np.radians(t_kat)) * np.sin(np.radians(d_kat)) - np.sin(np.radians(48)) * np.cos(np.radians(d_kat)))
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
    
    pozorovane_corrected = np.column_stack((t_corr, d_corr))

    
    return pozorovane_corrected

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

def plot_residuals_comparison(res_before, res_after):
    """ Vykreslí dva scatterploty vedľa seba pre porovnanie pred a po korekcii. """
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    def plot_scatter(ax, data, title):
        """ Pomocná funkcia na vykreslenie scatterplotu s indexami a kružnicami. """
        x, y = data
        ax.scatter(x, y, color='blue', alpha=0.7)

        # Pridanie indexov k bodom
        for i, (xi, yi) in enumerate(zip(x, y), start=1):
            ax.text(xi, yi, str(i), fontsize=9, ha='right', va='bottom', color='red')

        # Pridanie kružníc znázorňujúcich uhly
        circle1 = plt.Circle((0, 0), 0.5, color='r', fill=False, label='0.5°')
        circle2 = plt.Circle((0, 0), 0.25, color='g', fill=False, label='0.25°')
        circle3 = plt.Circle((0, 0), 0.05, color='b', fill=False, label='0.05°')

        for circle in [circle1, circle2, circle3]:
            ax.add_patch(circle)

        # Nastavenie rovnakých limitov pre obe osy
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')

        # Konfigurácia grafu
        ax.set_xlabel("X súradnice")
        ax.set_ylabel("Y súradnice")
        ax.set_title(title)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Legenda
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='0.5°', markeredgecolor='r', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='0.25°', markeredgecolor='g', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='0.05°', markeredgecolor='b', markersize=15)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    # Vykreslenie scatterplotov
    plot_scatter(axs[0], res_before, "Pred korekciou")
    plot_scatter(axs[1], res_after, "Po korekcii")

    # Úprava rozloženia
    plt.tight_layout()
    plt.show()

def main():
    # Počiatočné odhady korekčných faktorov (9 neznámych parametrov)
    initial_params = np.zeros(9)
    
    while True:
        print("\nWould you like to use real or simulated data?:")
        print("1 - Load data from a file")
        print("2 - Simulate observational data")
        print("3 - Close")

        choice = input("Input: ").strip()

        if choice == "3":
            print("Closing...")
            exit(0)
            break

        if choice not in ["1", "2"]:
            print("Invalid choice, please try again")
            continue

        try:
            n_stars = int(input("How many observations would you like to use for calculation of correction parameters?: ").strip())
            if n_stars <= 0:
                print("Nuber of observations has to be greater than 0")
                continue
        except ValueError:
            print("Invalid input")
            continue

        mode = "file" if choice == "1" else "simulate"

        katalog, pozorovane = load_data(mode, n_stars)
        print(f"{n_stars} observations loaded.")

        try:
            print(n_stars, mode)
            katalog, pozorovane = load_data(mode, n_stars)
            print(f"{n_stars} observations loaded.")
            break
        except Exception as e:
            print(f"Error: {e}")

    latitude = get_latitude()

    optimal_params, _ = leastsq(error_function, initial_params, args=(katalog, pozorovane, latitude))
    pozorovane_corrected = calculate_corrected_coords(optimal_params, pozorovane, 48)

    #Výpis prvej tabulky súradnice
    print(f"{'HA':>10}\t\t{'DEC':>10}\t\t\t{'HA_o':>10}\t\t\t{'DEC_o':>10}\t\t\t{'HA_c':>10}\t\t\t{'DEC_c':>10}")
    for (ha, dec), (ha_o, dec_o), (ha_c, dec_c) in zip(katalog, pozorovane, pozorovane_corrected):
        print(f"{ha:10.4f}\t\t{dec:10.4f}\t\t\t{ha_o:10.4f}\t\t\t{dec_o:10.4f}\t\t\t{ha_c:10.4f}\t\t\t{dec_c:10.4f}")
    print("\n")


    with open("data.txt", "w") as f:
        f.write(f"{'HA':>15}\t{'DEC':>15}\t{'HA_corrected':>15}\t{'DEC_corrected':>15}\n")
        for (ha, dec), (ha_c, dec_c) in zip(katalog, pozorovane_corrected):
            f.write(f"{ha:15.4f}\t{dec:15.4f}\t{ha_c:15.4f}\t{dec_c:15.4f}\n")


    res_before, res_after = compute_residuals(katalog, pozorovane, pozorovane_corrected)
    res_ha_before, res_dec_before = res_before
    res_ha_after, res_dec_after = res_after

    #Výpis druhej tabulky rezidua
    print(f"\t\t\t\t\t\t\t{'RES_HA':>10}\t\t\t{'RES_DEC':>10}\t\t\t{'RES_HA_c':>10}\t\t\t{'RES_DEC_c':>10}")
    for ha, dec, ha_c, dec_c in zip(res_ha_before, res_dec_before, res_ha_after, res_dec_after):
        print(f"\t\t\t\t\t\t\t{ha:10.4f}\t\t\t{dec:10.4f}\t\t\t{ha_c:10.4f}\t\t\t{dec_c:10.4f}")


    plot_residuals_comparison(res_before, res_after)
    #plot_polar_comparison(katalog, pozorovane, pozorovane_corrected)
    plot_cartesian_comparison_with_error_components(katalog, pozorovane, pozorovane_corrected)

    # Výpis výsledkov
    ZH_c, ZD_c, CO_c, NP_c, MA_c, ME_c, TF_c, DF_c, FO_c = optimal_params
    ZH_t, ZD_t, CO_t, NP_t, MA_t, ME_t, TF_t, DF_t, FO_t = true_params

    print(f"Optimalizované korekčné faktory montáže pri {n_stars} pozorovaniach:")

    print(f"ZH (nulový bod hodinového uhla): δ_ZH={abs(ZH_c - ZH_t):.4f}")
    print(f"ZD (nulový bod deklinácie): δ_ZD={abs(ZD_c - ZD_t):.4f}")
    print(f"CO (kolimácia): δ_CO={abs(CO_c - CO_t):.4f}")
    print(f"NP (nekolmosť osí): δ_NP={abs(NP_c - NP_t):.4f}")
    print(f"MA (chyba vyrovnania E-W): δ_MA={abs(MA_c - MA_t):.4f}")
    print(f"ME (chyba vyrovnania N-S): δ_ME={abs(ME_c - ME_t):.4f}")
    print(f"TF (priehyb tubusu): δ_TF={abs(TF_c - TF_t):.4f}")
    print(f"DF (chyba deklinácie): δ_DF={abs(DF_c - DF_t):.4f}")
    print(f"FO (chyba vidlice): δ_FO={abs(FO_c - FO_t):.4f}")
    print("\n")


if __name__ == "__main__":
    main()