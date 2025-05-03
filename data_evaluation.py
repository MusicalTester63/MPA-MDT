import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from geopy.geocoders import Nominatim
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from geopy.geocoders import Nominatim


#               ZH   ZD    CO    NP    MA    ME    TF     DF    FO
#true_params = [0.1, 0.1, -0.05, 0.01, -0.1, 0.1, 0.08, -0.01, 0]  #Simulacia
true_params = [-0.1424, -0.0141, -0.0583, 0.002, -0.2551, 0.1394, 0.0144, 0.2313, -0.2458]  #Približne skutočné 

def get_latitude():
    try:
        geolocator = Nominatim(user_agent="geo_locator")
        location_string = "Važec,slovakia"
        location = geolocator.geocode(location_string)
        if location:
            print(f"Located {location_string} on latitude {location.latitude}")
            return location.latitude
        else:
            return 48.0
    except Exception as e:
        print(f"Chyba pri geolokácii: {e}")
        return 48.0
    
def ra_to_ha(ra, observing_time_str):
    """
    Prepočet RA na HA pre lokalitu načítanú cez get_latitude().

    Args:
        ra (float): Rektascenzia v stupňoch (0 - 360).
        observing_time_str (str): Čas pozorovania v ISO formáte, napr. '2025-04-06T22:00:00'.

    Returns:
        float: Hodinový uhol v stupňoch (-180° až +180°).
    """
    # Získanie dynamickej šírky
    latitude = get_latitude()

    # Lokalita (šírka dynamická, ostatné pevné)
    observing_location = EarthLocation(lat=latitude*u.deg, lon=17.1077*u.deg, height=150*u.m)
    
    # Čas pozorovania
    observing_time = Time(observing_time_str)

    # Výpočet miestneho hviezdneho času (LST)
    lst = observing_time.sidereal_time('apparent', longitude=observing_location.lon)

    # Prevod RA na SkyCoord
    coords = SkyCoord(ra=ra*u.deg, dec=0*u.deg, frame='icrs')  # DEC tu nehrá rolu

    # Výpočet HA
    ha = (lst - coords.ra).wrap_at(12 * u.hour)
    ha_deg = ha.to(u.deg).value

    return ha_deg

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
        katalog, pozorovane = simulate_observation_gaia(n_stars)
        

    elif mode == "file":
        data = np.loadtxt("./data/test_data_sim.txt", skiprows=1)  # Preskočí hlavičku
        if len(data) < n_stars:
            raise ValueError(f"File contains only {len(data)} rows of data. You required {n_stars}.")
        katalog = data[:n_stars, :2]  # HA a DEC (katalógové súradnice)
        pozorovane = data[:n_stars, 2:]  # HA_PNT a DEC_PNT (pozorované súradnice)

    else:
        raise ValueError('Invalid mode. Please use ".')

    return katalog, pozorovane

def export_coords_observation(coordinate_array, file_name, header):
    with open(file_name, "w") as f:
        # Zapíš hlavičku
        header_line = "\t".join(f"{col:>15}" for col in header)
        f.write(header_line + "\n")
        
        # Zapíš dáta
        for row in coordinate_array:
            row_line = "\t".join(f"{val:15.4f}" for val in row)
            f.write(row_line + "\n")

def simulate_observation_gaia(n_stars):
    """
    Vyber náhodné hviezdy z Gaia katalógu s RA, DEC.

    Zadefinuj čas a miesto pozorovania.

    Spočítaj HA pre každú hviezdu podľa daného času a miesta.

    Vytvor pole [HA, DEC] pre každú hviezdu.

    Pridaj chybu do týchto údajov (napr. cez apply_errors).

    Výstup bude napodobňovať reálne "merané" dáta.
    """
    adql_query = f"""
    SELECT TOP {n_stars} source_id, ra, dec
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag < 15
      AND dec > -5
      AND dec < 75
    ORDER BY random_index
    """

    job = Gaia.launch_job(adql_query)
    results = job.get_results()

    ra = np.array(results['ra'])
    dec = np.array(results['dec'])

    observing_data = np.column_stack((ra, dec))
    export_coords_observation(observing_data, "target_coordinates_catalog_RA_DEC.txt", ["RA", "DEC"])
 
    ha_deg = ra_to_ha(ra, "2025-04-19T20:00:00")

    # Tvorba simulovaného katalógu
    katalog = np.column_stack((ha_deg, dec))

    export_coords_observation(katalog, "target_coordinates_catalog_HA_DEC.txt", ["HA", "DEC"])

    # Pridanie meracích chýb
    pozorovane = apply_errors(true_params, katalog, n_stars, noise_scale=0.01)

    return katalog, pozorovane

def apply_errors(params, catalogue, n_stars, latitude_deg=get_latitude(), noise_scale=0.1):
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params
    HA_kat, DEC_kat = catalogue[:, 0], catalogue[:, 1]
    
    # Aplikácia systémových chýb
    HA_poz = (
        HA_kat 
        + ZH 
        + CO / np.cos(np.radians(DEC_kat)) 
        + NP * np.tan(np.radians(DEC_kat))
        - MA * np.cos(np.radians(HA_kat)) * np.tan(np.radians(DEC_kat))
        + ME * np.sin(np.radians(HA_kat)) * np.tan(np.radians(DEC_kat))
        + TF * np.cos(np.radians(latitude_deg)) * np.sin(np.radians(HA_kat)) / np.cos(np.radians(DEC_kat))
        - DF * (np.cos(np.radians(latitude_deg)) * np.cos(np.radians(HA_kat)) + np.sin(np.radians(latitude_deg)) * np.tan(np.radians(DEC_kat)))
    )
    
    DEC_poz = (
        DEC_kat 
        + ZD 
        + MA * np.sin(np.radians(HA_kat)) 
        + ME * np.cos(np.radians(HA_kat))
        + TF * (np.cos(np.radians(latitude_deg)) * np.cos(np.radians(HA_kat)) * np.sin(np.radians(DEC_kat)) 
        - np.sin(np.radians(latitude_deg)) * np.cos(np.radians(DEC_kat)))
        + FO * np.cos(np.radians(HA_kat))
    )
    
    HA_poz += np.random.normal(0, noise_scale, n_stars)
    DEC_poz += np.random.normal(0, noise_scale, n_stars)
    
    return np.column_stack((HA_poz, DEC_poz))

def plot_cartesian_comparison_with_error_components(catalogue, observed, observed_corrected):
    """
    Vykreslí karteziánsky graf pre catalogue, observed a observed_corrected,
    s čiarami znázorňujúcimi chyby v HA a DEC.

    Parametre:
    ----------
    catalogue : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty z katalógu.
    observed : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní pred korekciou.
    observed_corrected : numpy.ndarray
        Pole tvaru (n, 2) obsahujúce HA a DEC hodnoty pozorovaní po korekcii.
    """
    ha_catalogue, dec_catalogue = catalogue[:, 0], catalogue[:, 1]
    ha_observed, dec_observed = observed[:, 0], observed[:, 1]
    ha_corrected, dec_corrected = observed_corrected[:, 0], observed_corrected[:, 1]

    plt.figure(figsize=(12, 8))

    plt.scatter(ha_catalogue, dec_catalogue, color='blue', label='Katalóg', alpha=0.6)

    plt.scatter(ha_observed, dec_observed, color='red', label='Pozorované (pred korekciou)', alpha=0.6)
    for ha_k, dec_k, ha_p, dec_p in zip(ha_catalogue, dec_catalogue, ha_observed, dec_observed):
        
        plt.plot([ha_k, ha_p], [dec_k, dec_k], color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Chyba HA' if ha_k == ha_catalogue[0] else "")
        
        plt.plot([ha_p, ha_p], [dec_k, dec_p], color='purple', linestyle='--', linewidth=1, alpha=0.6, label='Chyba DEC' if ha_k == ha_catalogue[0] else "")
        
        plt.text(ha_p, dec_k, f"HA: {ha_p - ha_k:.2f}", color='orange', fontsize=8, ha='left', va='bottom')
        plt.text(ha_p, dec_p, f"DEC: {dec_p - dec_k:.2f}", color='purple', fontsize=8, ha='left', va='bottom')

    plt.scatter(ha_corrected, dec_corrected, color='green', label='Pozorované (po korekcii)', alpha=0.6)
    for ha_k, dec_k, ha_c, dec_c in zip(ha_catalogue, dec_catalogue, ha_corrected, dec_corrected):
        
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

def compute_residuals(catalogue, observed, observed_corrected):
    # Rezíduá pred korekciou
    res_ha_before = observed[:, 0] - catalogue[:, 0]
    res_dec_before = observed[:, 1] - catalogue[:, 1]
    
    # Rezíduá po korekcii
    res_ha_after = observed_corrected[:, 0] - catalogue[:, 0]
    res_dec_after = observed_corrected[:, 1] - catalogue[:, 1]
    
    return (res_ha_before, res_dec_before), (res_ha_after, res_dec_after)

def error_function(params, catalogue, observed, latitude_deg):
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params

    t_catalogue, d_catalogue = catalogue[:, 0], catalogue[:, 1]
    t_observed, d_observed = observed[:, 0], observed[:, 1]
    
    t_corr = t_observed - (t_catalogue + ZH + CO / np.cos(np.radians(d_catalogue)) + NP * np.tan(np.radians(d_catalogue))
                     - MA * np.cos(np.radians(t_catalogue)) * np.tan(np.radians(d_catalogue))
                     + ME * np.sin(np.radians(t_catalogue)) * np.tan(np.radians(d_catalogue))
                     + TF * np.cos(np.radians(latitude_deg)) * np.sin(np.radians(t_catalogue)) / np.cos(np.radians(d_catalogue))
                     - DF * (np.cos(np.radians(latitude_deg)) * np.cos(np.radians(t_catalogue)) + np.sin(np.radians(latitude_deg)) * np.tan(np.radians(d_catalogue)) ))
    
    d_corr = d_observed - (d_catalogue + ZD + MA * np.sin(np.radians(t_catalogue)) + ME * np.cos(np.radians(t_catalogue))
                     + TF * (np.cos(np.radians(latitude_deg)) * np.cos(np.radians(t_catalogue)) * np.sin(np.radians(d_catalogue)) - np.sin(np.radians(latitude_deg)) * np.cos(np.radians(d_catalogue)))
                     + FO * np.cos(np.radians(t_catalogue)))
    
    return np.concatenate((t_corr, d_corr))

def calculate_corrected_coords(params, observed, latitude_deg):
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params
    t_observed, d_poz = observed[:, 0], observed[:, 1]
    
    # Korekcia: Od nameraných hodnôt ODČÍTAME chyby montáže
    t_corr = t_observed - (ZH + CO / np.cos(np.radians(d_poz)) + NP * np.tan(np.radians(d_poz))
                     - MA * np.cos(np.radians(t_observed)) * np.tan(np.radians(d_poz))
                     + ME * np.sin(np.radians(t_observed)) * np.tan(np.radians(d_poz))
                     + TF * np.cos(np.radians(latitude_deg)) * np.sin(np.radians(t_observed)) / np.cos(np.radians(d_poz))
                     - DF * (np.cos(np.radians(latitude_deg)) * np.cos(np.radians(t_observed)) + np.sin(np.radians(latitude_deg)) * np.tan(np.radians(d_poz)) )
    )
    
    d_corr = d_poz - (ZD + MA * np.sin(np.radians(t_observed)) + ME * np.cos(np.radians(t_observed))
                     + TF * (np.cos(np.radians(latitude_deg)) * np.cos(np.radians(t_observed)) * np.sin(np.radians(d_poz)) 
                     - np.sin(np.radians(latitude_deg)) * np.cos(np.radians(d_poz)))
                     + FO * np.cos(np.radians(t_observed)))
    
    observed_corrected = np.column_stack((t_corr, d_corr))

    return observed_corrected

def correct_target_coordinates(params, catalog_coords, latitude_deg):
    
    ZH, ZD, CO, NP, MA, ME, TF, DF, FO = params
    HA_cat, DEC_cat = catalog_coords[:, 0], catalog_coords[:, 1]

    # Pridáme chyby ako "anti-chyby" aby teleskop mieril správne
    HA_corr = HA_cat - (
        ZH
        + CO / np.cos(np.radians(DEC_cat))
        + NP * np.tan(np.radians(DEC_cat))
        - MA * np.cos(np.radians(HA_cat)) * np.tan(np.radians(DEC_cat))
        + ME * np.sin(np.radians(HA_cat)) * np.tan(np.radians(DEC_cat))
        + TF * np.cos(np.radians(latitude_deg)) * np.sin(np.radians(HA_cat)) / np.cos(np.radians(DEC_cat))
        - DF * (np.cos(np.radians(latitude_deg)) * np.cos(np.radians(HA_cat)) + np.sin(np.radians(latitude_deg)) * np.tan(np.radians(DEC_cat)))
    )

    DEC_corr = DEC_cat - (
        ZD
        + MA * np.sin(np.radians(HA_cat))
        + ME * np.cos(np.radians(HA_cat))
        + TF * (np.cos(np.radians(latitude_deg)) * np.cos(np.radians(HA_cat)) * np.sin(np.radians(DEC_cat)) - np.sin(np.radians(latitude_deg)) * np.cos(np.radians(DEC_cat)))
        + FO * np.cos(np.radians(HA_cat))
    )

    corrected_targets = np.column_stack((HA_corr, DEC_corr))

    combined = np.column_stack((HA_cat, DEC_cat, HA_corr, DEC_corr))

    export_coords_observation(
        combined,
        "corrected_target_coordinates.txt",
        ["HA_cat", "DEC_cat", "HA_corr", "DEC_corr"]
    )

    return corrected_targets

def plot_residuals(res_before, res_after):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Hodinový uhol (HA)
    ax[0].scatter(range(len(res_before[0])), res_before[0], c='r', label='Pred korekciou')
    ax[0].scatter(range(len(res_after[0])), res_after[0], c='g', marker='x', label='Po korekcii')
    ax[0].set_title('Hour angle residuals (HA)')
    ax[0].set_ylabel('Odchýlka [°]')
    ax[0].legend()
    ax[0].grid(True)
    
    # Deklinácia (DEC)
    ax[1].scatter(range(len(res_before[1])), res_before[1], c='r', label='Pred korekciou')
    ax[1].scatter(range(len(res_after[1])), res_after[1], c='g', marker='x', label='Po korekcii')
    ax[1].set_title('Declination residuals (DEC)')
    ax[1].set_xlabel('Číslo hviezdy')
    ax[1].set_ylabel('Odchýlka [°]')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_scatter(ax, data, title):
    x, y = data
    ax.scatter(x, y, color='blue', alpha=0.7)

    # Pridanie indexov k bodom
    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        ax.text(xi, yi, str(i), fontsize=9, ha='right', va='bottom', color='red')

    # Pridanie kružníc znázorňujúcich uhly
    circle1 = plt.Circle((0, 0), 0.5, color='r', fill=False, label='0.5°')
    circle2 = plt.Circle((0, 0), 0.25, color='g', fill=False, label='0.25°')
    circle3 = plt.Circle((0, 0), 0.05, color='b', fill=False, label='0.05°')
    circle4 = plt.Circle((0, 0), 0.025, color='m', fill=False, label='0.025°')

    for circle in [circle1, circle2, circle3, circle4]:
        ax.add_patch(circle)

    x_lim = 0.65
    y_lim = 0.65

    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)

    d_x_max = abs(x_lim-x_max)
    d_x_min = abs(x_lim-x_min)
    d_y_max = abs(y_lim-y_max)
    d_y_min = abs(y_lim-y_min)

    # Nastavenie rovnakých limitov pre obe osy
    ax.set_xlim(-(x_min+d_x_min), x_max+d_x_max)
    ax.set_ylim(-(y_min+d_y_min), y_max+d_y_max)
    ax.set_aspect('equal')


    # Konfigurácia grafu
    ax.set_xlabel("Hour angle residuals (HA)[°]")
    ax.set_ylabel("Declination residuals (DEC)[°]")
    ax.set_title(title)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='0.5°', markeredgecolor='r', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='0.25°', markeredgecolor='g', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='0.05°', markeredgecolor='b', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='0.025°', markeredgecolor='m', markersize=15),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

def plot_residuals_comparison(res_before, res_after):
    """ Vykreslí dva scatterploty vedľa seba pre porovnanie pred a po korekcii. """
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Vykreslenie scatterplotov
    plot_scatter(axs[0], res_before, "Without correction")
    plot_scatter(axs[1], res_after, "With correction")

    # Úprava rozloženia
    plt.tight_layout()
    plt.show()





def calculate_rms_residuals(res_before, res_after):
    """
    Vypočíta RMS rezíduí pre HA a DEC pred a po korekcii v stupňoch a arcsekundách.
    
    Parametre:
        res_before (tuple): Rezíduá pred korekciou (res_ha_before, res_dec_before).
        res_after (tuple): Rezíduá po korekcii (res_ha_after, res_dec_after).
    
    Vráti:
        tuple: RMS hodnoty v tvare (rms_ha_before_deg, rms_dec_before_deg, 
               rms_ha_after_deg, rms_dec_after_deg,
               rms_ha_before_arcsec, rms_dec_before_arcsec,
               rms_ha_after_arcsec, rms_dec_after_arcsec)
    """
    res_ha_before, res_dec_before = res_before
    res_ha_after, res_dec_after = res_after
    
    # Výpočet RMS v stupňoch
    rms_ha_before_deg = np.sqrt(np.mean(res_ha_before**2))
    rms_dec_before_deg = np.sqrt(np.mean(res_dec_before**2))
    rms_ha_after_deg = np.sqrt(np.mean(res_ha_after**2))
    rms_dec_after_deg = np.sqrt(np.mean(res_dec_after**2))
    
    # Konverzia na oblúkové sekundy (1° = 3600″)
    rms_ha_before_arcsec = rms_ha_before_deg * 3600
    rms_dec_before_arcsec = rms_dec_before_deg * 3600
    rms_ha_after_arcsec = rms_ha_after_deg * 3600
    rms_dec_after_arcsec = rms_dec_after_deg * 3600
    
    return (rms_ha_before_deg, rms_dec_before_deg, 
            rms_ha_after_deg, rms_dec_after_deg,
            rms_ha_before_arcsec, rms_dec_before_arcsec,
            rms_ha_after_arcsec, rms_dec_after_arcsec)

def print_rms_results(rms_results):
    """
    Vypíše RMS rezíduá do konzoly s jednotkami v stupňoch aj arcsekundách.
    """
    (rms_ha_before_deg, rms_dec_before_deg, 
     rms_ha_after_deg, rms_dec_after_deg,
     rms_ha_before_arcsec, rms_dec_before_arcsec,
     rms_ha_after_arcsec, rms_dec_after_arcsec) = rms_results
    
    print("\n" + "="*80)
    print("RMS REZIDUÁ".center(80))
    print("="*80)
    print(f"{'':<20}{'Pred korekciou':<30}{'Po korekcii':<30}")
    print(f"{'HA (hodinový uhol)':<20}{rms_ha_before_deg:.6f}° ({rms_ha_before_arcsec:.2f}″){'':<10}{rms_ha_after_deg:.6f}° ({rms_ha_after_arcsec:.2f}″)")
    print(f"{'DEC (deklinácia)':<20}{rms_dec_before_deg:.6f}° ({rms_dec_before_arcsec:.2f}″){'':<10}{rms_dec_after_deg:.6f}° ({rms_dec_after_arcsec:.2f}″)")
    print("="*80 + "\n")














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

        try:
            katalog, pozorovane = load_data(mode, n_stars)
            print(f"{n_stars} observations loaded.")
            break
        except Exception as e:
            print(f"Error: {e}")

    latitude = get_latitude()

    optimal_params, _ = leastsq(error_function, initial_params, args=(katalog, pozorovane, latitude))

    pozorovane_corrected = calculate_corrected_coords(optimal_params, pozorovane, latitude)

    #Výpis prvej tabulky súradnice
    print(f"{'HA':>10}\t\t{'DEC':>10}\t\t\t{'HA_o':>10}\t\t\t{'DEC_o':>10}\t\t\t{'HA_c':>10}\t\t\t{'DEC_c':>10}")
    for (ha, dec), (ha_o, dec_o), (ha_c, dec_c) in zip(katalog, pozorovane, pozorovane_corrected):
        print(f"{ha:10.4f}\t\t{dec:10.4f}\t\t\t{ha_o:10.4f}\t\t\t{dec_o:10.4f}\t\t\t{ha_c:10.4f}\t\t\t{dec_c:10.4f}")
    print("\n")
    
    res_before, res_after = compute_residuals(katalog, pozorovane, pozorovane_corrected)
    res_ha_before, res_dec_before = res_before
    res_ha_after, res_dec_after = res_after

    #Výpis druhej tabulky rezidua
    print(f"\t\t\t\t\t\t\t{'RES_HA':>10}\t\t\t{'RES_DEC':>10}\t\t\t{'RES_HA_c':>10}\t\t\t{'RES_DEC_c':>10}")
    for ha, dec, ha_c, dec_c in zip(res_ha_before, res_dec_before, res_ha_after, res_dec_after):
        print(f"\t\t\t\t\t\t\t{ha:10.4f}\t\t\t{dec:10.4f}\t\t\t{ha_c:10.4f}\t\t\t{dec_c:10.4f}")


    # Vypočítaj a vypíš RMS rezíduá
    rms_results = calculate_rms_residuals(res_before, res_after)
    print_rms_results(rms_results)








    output_path = "./output"

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_scatter(ax, res_before, f"Without correction, n_stars={n_stars}")
    plt.savefig(f'{output_path}/sim_residuals_before.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_scatter(ax, res_after, f"With correction, n_stars={n_stars}")
    plt.savefig(f'{output_path}/sim_residuals_after.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


    plot_residuals_comparison(res_before, res_after)
    #plot_cartesian_comparison_with_error_components(katalog, pozorovane, pozorovane_corrected)
    # Výpis výsledkov
    
    ZH_c, ZD_c, CO_c, NP_c, MA_c, ME_c, TF_c, DF_c, FO_c = optimal_params
    ZH_t, ZD_t, CO_t, NP_t, MA_t, ME_t, TF_t, DF_t, FO_t = true_params


    if mode == "simulate":

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


    print(f"Optimalizované korekčné faktory montáže pri {n_stars} pozorovaniach:")

    print(f"ZH (nulový bod hodinového uhla): ZH={ZH_c:.4f}")
    print(f"ZD (nulový bod deklinácie): ZD={ZD_c:.4f}")
    print(f"CO (kolimácia): CO={CO_c:.4f}")
    print(f"NP (nekolmosť osí): NP={NP_c:.4f}")
    print(f"MA (chyba vyrovnania E-W): MA={MA_c:.4f}")
    print(f"ME (chyba vyrovnania N-S): ME={abs(ME_c):.4f}")
    print(f"TF (priehyb tubusu): TF={TF_c:.4f}")
    print(f"DF (chyba deklinácie): DF={DF_c:.4f}")
    print(f"FO (chyba vidlice): FO={FO_c:.4f}")
    print("\n")

    correct_target_coordinates(optimal_params, katalog, latitude)


if __name__ == "__main__":
    main()