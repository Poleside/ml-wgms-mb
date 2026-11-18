# ML4GM glacier mass-balance reconstruction

Reproduction code for manuscript v0.3.30, *Reconstructing annual and seasonal glacier mass balance across the global observational network, 1979–2025*. The workflow combines WGMS observations, Randolph Glacier Inventory 6.0 attributes, and ERA5-Land forcing to reconstruct annual, winter, and summer glacier-wide mass balance for the WGMS-linked glacier network.

## Set up the environment

```sh
conda env create -f environment.yml
conda activate ml4gm-tc
```

Run the commands below from the repository root.

## Data sources

| Source | Version and input files |
| --- | --- |
| WGMS Fluctuations of Glaciers | [2026-02-10 release](https://doi.org/10.5904/wgms-fog-2026-02-10). Place `glacier.csv` and `mass_balance.csv` in `data/DOI-WGMS-FoG-2026-02-10/data/`. |
| Randolph Glacier Inventory | [RGI 6.0, NSIDC-0770](https://doi.org/10.7265/4M1F-GD79). Place the 19 regional attribute CSVs (`01_rgi60_*.csv` through `19_rgi60_*.csv`) in `data/RGI6/`. For Fig. 2, add `00_rgi60_O1Regions.shp` and its shapefile sidecar files there. |
| ERA5-Land | Use [Earth Engine Monthly Aggregated](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR), collection `ECMWF/ERA5_LAND/MONTHLY_AGGR`. See the [dataset article](https://doi.org/10.5194/essd-13-4349-2021) and the collection's terms of use. |
| Natural Earth | Cartopy can fetch the 1:110 million [physical](https://www.naturalearthdata.com/downloads/110m-physical-vectors/) and [cultural](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/) map layers. For offline use, place the standard `ne_110m_*` shapefiles and sidecars under `figures/cartopy_data/shapefiles/natural_earth/`, in `physical/` or `cultural/`. |

## Run the analysis

Open the notebooks in `code/` in Jupyter or VS Code with the `ml4gm-tc` Python environment as the kernel. Run each from top to bottom, starting with `01_Dataset.ipynb`. To run a notebook from a terminal, pass its path to IPython's `%run`, for example:

```sh
python -m IPython --HistoryManager.enabled=False -c "%run code/01_Dataset.ipynb"
```

On a first run without climate exports, notebook 01 stops after writing `data/raw/era5land_gee/wgms_global_glacier_points_for_gee.csv`. Upload that table as an asset in your Earth Engine project, set `POINT_ASSET` in [the export script](code/gee_export_era5land.js), and export 1979–2025. If exporting in batches, use nonoverlapping year ranges or regions that together cover the study inputs. Place the downloaded CSVs in `data/raw/era5land_gee/exports/` and rerun 01. If you already have the normalized table, place it at `data/raw/era5land_gee/wgms_global_era5land_1979_2025.csv` instead.

The manuscript uses 1979–2025 forcing; some early seasonal windows use only the available months. For a separate sensitivity run, a 1978 export can be placed in `data/raw/era5land_gee/extension/`. Rerun from step 01 and repeat hyperparameter selection for that run.

Once step 01 completes, run notebooks 02, 03, 04, 05, 06, 08, and 09 in that order. Then run `python code/14_regional_change_attribution.py`.

| Step | Outputs for the manuscript |
| --- | --- |
| 01 | WGMS–RGI links, monthly forcing, annual target matrix, and RGI regional area weights |
| 02 | Annual and seasonal model-ready matrices and reported seasonal windows |
| 03 | Fitted XGBoost models, fold-level predictions, validation metrics, and predictor gains |
| 04 | Seasonal-window withholding and balance-prediction sensitivity |
| 05 | Annual, winter, and summer reconstructions and observation/model source flags |
| 06 | Glacier-year and aggregate empirical uncertainty |
| 08 | Training-mean and ridge-regression benchmarks on the manuscript splits |
| 09 | Predictor-group permutation diagnostics |
| 14 | Five-region winter and summer fitted changes and signed predictor-group contributions |

The initial model training includes hyperparameter search and cross-validation and can take substantially longer than the other steps.

Set the environment variable `ML4GM_FORCE_PARAM_SEARCH` to `1` before running step 03 after any change to the training records or predictors. This recomputes hyperparameter selection for all three targets even when parameter files already exist.

The notebooks write intermediates under `data/`, `models/`, and `results/`. To keep inputs and results outside the package, set `ML4GM_ROOT` to an existing directory before starting the notebook kernel or IPython. Pass the same directory to the regional analysis and figure scripts with `--root`. The annual product covers 317 glaciers. Annual summaries use the same 316 glaciers in every year after excluding the single glacier from RGI region 19. The seasonal samples and the 197-glacier analysis in Fig. 9 have their own coverage criteria.

## Generate manuscript figures

Figure 1 uses Sharp to render its SVG. Install Arial and Node.js 20.9 or later, then run `npm install` from the package directory before generating Figure 1. This figure exports an editable vector SVG, a 3200 × 1680 pixel PNG at 300 dpi, and a raster PDF.

```sh
python figures/fig05_reconstructed_mass_balance.py
```

Each script in `figures/` with a name from `fig01_*.py` to `fig09_*.py` runs independently and writes its PNG, PDF and any source tables to `figures/exports/`. Use `--root` for an external data directory and `--output-dir` for a different export location. PNGs default to 300 dpi. Figure 2 retains the manuscript map's original pixel dimensions at 300 dpi; no separate command-line setting is needed. Figure 9 checks that the signed group contributions sum to the fitted change.

For v0.3.30, the annual, winter and summer training sets contain 5,465, 2,519 and 2,488 records, respectively. The fixed RGI 6.0 area is used as a predictor and for glacier-area weighting. Figure 7's cumulative annual balances are −9.04 m w.e. for the study sample and −15.40 m w.e. after regional area reweighting. The shaded bands in Figs. 5 and 7 show one empirical standard uncertainty on each side.

## Citation

Cite *ML4GM Glacier Mass-Balance Reconstruction*, version 0.1.0, using [CITATION.cff](CITATION.cff). The code is released under the [MIT License](LICENSE). Third-party inputs retain their providers' licences and should be cited separately.
