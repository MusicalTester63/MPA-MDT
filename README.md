# Telescope Mount Error Correction Script

## Overview

This Python script analyzes and corrects systematic errors in equatorial telescope mounts by comparing observed star positions with catalogued positions. It implements a comprehensive physical model of common telescope mount errors and provides tools for:

1. Simulating observations with configurable error parameters
2. Processing real observation data
3. Calculating optimal correction parameters using least squares fitting
4. Evaluating correction effectiveness through statistical and visual analysis
5. Generating corrected coordinates for future observations

## Key Features

- **Physical Error Modeling**: Implements 9-parameter physical model of telescope mount errors
- **Flexible Data Input**: Works with both simulated (Gaia DR3 catalog) and real observation data
- **Statistical Optimization**: Uses Levenberg-Marquardt least squares algorithm for parameter fitting
- **Comprehensive Visualization**: Generates multiple diagnostic plots in PDF format
- **Practical Outputs**: Produces ready-to-use corrected coordinates for telescope control systems
- **Geolocation Integration**: Automatically determines observer latitude or allows manual override

## Dependencies

### Core Requirements
- Python 3.7+  
- numpy (for numerical computations)
- scipy (for optimization algorithms)
- matplotlib (for visualization)
- astropy (for astronomical calculations)

### Installing Dependencies

To install all necessary libraries, run:
```bash
pip install -r requirements.txt
```

## Data Requirements

### For Real Observations (Option 1)
Create a text file `./data/test_data_sim.txt` containing:
- Tab or space separated values
- First row is header (will be skipped)
- Each subsequent row contains:
  - Catalog Hour Angle (HA) in degrees
  - Catalog Declination (DEC) in degrees
  - Observed Hour Angle (HA_PNT) in degrees
  - Observed Declination (DEC_PNT) in degrees

Example format:
```
HA_cat    DEC_cat    HA_obs    DEC_obs
12.345    45.678    12.355    45.688
...
```

### For Simulations (Option 2)
No input files needed - the script will:
1. Query Gaia DR3 catalog for random stars (G < 15 mag)
2. Calculate their positions for specified observation time
3. Apply systematic errors based on configurable parameters

## How to Use

### Basic Operation
1. Run the script:
   ```bash
   python data_evaluation.py
   ```
2. Follow the interactive menu prompts

### Menu Options
```
Would you like to use real or simulated data?:
1 - Load data from a file
2 - Simulate observational data
3 - Close
```

#### Option 1: Real Data Processing
1. Prepare your observation data file
2. Select option `1`
3. Enter number of observations to use (must be ≤ available data points)
4. Script will:
   - Load and process the data
   - Calculate optimal correction parameters
   - Generate diagnostic plots and output files

#### Option 2: Simulation Mode
1. Select option `2`
2. Enter number of stars to simulate (typically 10-100)
3. Script will:
   - Query Gaia DR3 for star positions
   - Simulate observations with known errors
   - Attempt to recover error parameters
   - Validate correction effectiveness

### Advanced Configuration

1. **Location Settings**:
   - Modify `location_string` in `get_latitude()` function
   - Default: "Važec,slovakia" (falls back to lat=48.0° if geolocation fails)

2. **Observation Parameters**:
   - Change observation time in `simulate_observation_gaia()` (default: "2025-04-19T20:00:00")
   - Adjust noise level via `noise_scale` parameter in `apply_errors()`

3. **Error Model**:
   - Modify `true_params` array to change simulation error values
   - [ZH, ZD, CO, NP, MA, ME, TF, DF, FO] in degrees

## Output Files

1. **Coordinate Files**:
   - `target_coordinates_catalog_RA_DEC.txt`: Original RA/DEC from catalog
   - `target_coordinates_catalog_HA_DEC.txt`: Calculated HA/DEC coordinates
   - `catalog_vs_observed_simulation.txt`: Comparison of catalog vs observed positions
   - `corrected_target_coordinates.txt`: Final corrected coordinates for telescope use

2. **Diagnostic Plots** (in ./output/ directory):
   - `cartesian_comparison.pdf`: HA/DEC comparison before/after correction
   - `sim_residuals_before.pdf`: Residuals without correction
   - `sim_residuals_after.pdf`: Residuals after correction

## Error Model Details

The script models these physical telescope errors:

| Code | Error Type                  | Mathematical Form                      |
|------|-----------------------------|----------------------------------------|
| ZH   | Hour Angle Zero Point       | Constant offset in HA                  |
| ZD   | Declination Zero Point      | Constant offset in DEC                 |
| CO   | Collimation Error           | ∝ 1/cos(DEC)                           |
| NP   | Non-Perpendicularity        | ∝ tan(DEC)                             |
| MA   | Misalignment (E-W)          | ∝ cos(HA)tan(DEC)                      |
| ME   | Misalignment (N-S)          | ∝ sin(HA)tan(DEC)                      |
| TF   | Tube Flexure                | ∝ sin(HA)/cos(DEC)                     |
| DF   | Declination Flexure         | ∝ cos(HA) + tan(DEC) terms             |
| FO   | Fork Flexure                | ∝ cos(HA)                              |

## Interpreting Results

1. **Numerical Output**:
   - RMS residuals (in degrees and arcseconds)
   - Optimized correction parameters
   - For simulations: parameter recovery accuracy

2. **Diagnostic Plots**:
   - **Residual Scatter Plots**: Points should cluster near (0,0) after correction
   - **Cartesian Comparison**: Shows HA/DEC offsets visually
   - **Error Circles**: 0.5°, 0.25°, 0.05°, and 0.025° reference circles

3. **Success Metrics**:
   - Typical RMS after correction should be < 0.05° (3 arcminutes)
   - Excellent performance: < 0.01° (36 arcseconds)

## Troubleshooting

### Common Issues

1. **Data Loading Errors**:
   - Verify file exists at `./data/test_data_sim.txt`
   - Check for consistent column formatting
   - Ensure sufficient data points for requested analysis

2. **Gaia Query Failures**:
   - Check internet connection
   - Verify `astroquery` installation
   - May need to adjust ADQL query in `simulate_observation_gaia()`

3. **Geolocation Issues**:
   - Falls back to latitude 48.0° if geolocation fails
   - Can hardcode location in `get_latitude()`

4. **Optimization Problems**:
   - Ensure sufficient stars (minimum ~10, ideally 20+)
   - Stars should cover wide range of HA/DEC
   - May need better initial parameter guesses

## Example Workflows

### For Telescope Operators
1. Collect 20+ observations across the sky
2. Format data according to requirements
3. Run script with real data option
4. Apply correction parameters to telescope system
5. Verify improved pointing with new observations

### For Method Development
1. Run simulations with known parameters
2. Verify parameter recovery accuracy
3. Test with different noise levels
4. Experiment with star distributions
5. Validate statistical properties

## License & Attribution

This project is open-source under MIT License. When using this code in research, please cite the original authors and acknowledge the use of Gaia data.

## Support

For questions or issues, please open an issue on the project repository or contact the maintainers.