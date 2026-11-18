"""Decompose fitted regional seasonal balance changes for Figure 9.

Run with the ML4GM environment. Existing products/models are never overwritten.
Season-window ensemble members are reconstructed from stored window labels;
their predictions must reproduce the published product before attribution.
"""
from pathlib import Path
import argparse
import json
from os import environ
import re

import dill
import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
STATE = ['t2m', 'rsn', 'sde', 'tsn', 'fal', 'sp']
FLOW = ['sf', 'smlt', 'slhf', 'ssr', 'strd', 'sshf', 'ssrd', 'tp']
STATIC = ['Area', 'CenLon', 'CenLat', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Aspect', 'Lmax']
GROUPS = {
    'Snow / precipitation': ['tp', 'sf', 'rsn', 'sde'],
    'Temperature / melt': ['t2m', 'smlt', 'tsn'],
    'Radiation / flux / pressure': ['fal', 'slhf', 'sshf', 'ssr', 'ssrd', 'strd', 'sp'],
    'Geometry / interactions': STATIC,
}
SEASONS = ('winter', 'summer')


def read_product(root, season):
    return pd.read_csv(root / f'results/reconstruction/{season}_reconstruction_product_1979_2025_uncertainty.csv')


def period_delta(frame, columns):
    """Per glacier late-minus-early, each calendar year has equal weight."""
    frame = frame.copy()
    frame['period'] = np.where(frame.year.le(1999), 'early', 'late')
    p = frame.groupby(['RGIId', 'period'])[columns].mean().unstack('period')
    result = pd.DataFrame({c: p[(c, 'late')] - p[(c, 'early')] for c in columns})
    return result.join(frame.groupby('RGIId')[['O1Region', 'Area']].first()).reset_index()


def regional_summary(glaciers, columns):
    rows = []
    for region, g in glaciers.groupby('O1Region'):
        a = g.Area.to_numpy(float)
        largest = int(np.argmax(a))
        rng = np.random.default_rng(4200 + int(region))
        draws = rng.integers(0, len(g), (500, len(g)))
        for col in columns:
            v = g[col].to_numpy(float)
            boot = np.sum(v[draws] * a[draws], axis=1) / np.sum(a[draws], axis=1)
            loo = [(np.sum(v*a)-v[i]*a[i])/(a.sum()-a[i]) for i in range(len(g))] if len(g)>1 else [np.nan]
            rows.append(dict(O1Region=int(region), term=col, delta=float(np.average(v, weights=a)),
                             bootstrap_low=float(np.quantile(boot, .025)), bootstrap_high=float(np.quantile(boot, .975)),
                             leave_largest_out=float(loo[largest]), leave_one_out_min=float(np.nanmin(loo)),
                             leave_one_out_max=float(np.nanmax(loo)), n_glaciers=len(g),
                             maximum_area_share=float(a.max()/a.sum()), effective_glaciers=float(a.sum()**2/np.sum(a*a))))
    return pd.DataFrame(rows)


def rebuild_records(product, meta, era):
    """Expand stored windows; reproduce means of states and sums of fluxes."""
    members = []
    for row in product.itertuples(index=False):
        labels = str(row.window_pair).split('|')
        for j, label in enumerate(labels):
            match = re.fullmatch(r'([+-]?\d+):(\d+)->([+-]?\d+):(\d+)', label)
            if not match:
                raise ValueError(f'Cannot recover stored window {label}')
            so, sm, eo, em = map(int, match.groups())
            members.append(dict(RGIId=row.RGIId, year=int(row.year), O1Region=int(row.O1Region),
                                member_index=j, season_start_month=sm, season_end_month=em,
                                season_start_year_offset=so, season_end_year_offset=eo))
    records = pd.DataFrame(members)
    records['record_id'] = np.arange(len(records))
    months = []
    for r in records.itertuples(index=False):
        start = pd.Period(year=r.year+r.season_start_year_offset, month=r.season_start_month, freq='M')
        end = pd.Period(year=r.year+r.season_end_year_offset, month=r.season_end_month, freq='M')
        for p in pd.period_range(start, end, freq='M'):
            months.append((r.record_id, r.RGIId, p.year, p.month))
    months = pd.DataFrame(months, columns=['record_id', 'RGIId', 'year', 'month'])
    joined = months.merge(era, on=['RGIId','year','month'], how='inner', validate='many_to_one')
    grouped = joined.groupby('record_id')
    # Match the original per-window Series reductions exactly. Compensated
    # groupby sums can differ at a float32 tree split even by 1e-16.
    climate = grouped[STATE].agg(lambda x: x.mean()).join(
        grouped[FLOW].agg(lambda x: x.sum(min_count=1))).reset_index()
    records = records.merge(climate, on='record_id', how='inner', validate='one_to_one')
    records = records.merge(meta[['RGIId']+STATIC], on='RGIId', how='left', validate='many_to_one')
    return records


def supported_regions(payload):
    training = pd.DataFrame({'region': payload['o1region_all'], 'glacier': payload['rgiid_all']})
    support = training.groupby('region').glacier.agg(records='size', glaciers='nunique')
    return set(support.index[support.records.ge(100) & support.glaciers.ge(10)])


def run(root):
    out = root / 'results/analysis'
    out.mkdir(exist_ok=True, parents=True)
    annual, winter, summer = [read_product(root, season) for season in ('annual', *SEASONS)]
    keys = ['RGIId', 'year']
    aligned = annual[keys + ['O1Region', 'Area', 'annual_recon']].merge(
        winter[keys + ['winter_recon', 'winter_global_pred']], on=keys, how='inner', validate='one_to_one',
    ).merge(summer[keys + ['summer_recon', 'summer_global_pred']], on=keys, how='inner', validate='one_to_one')
    values = ['annual_recon', 'winter_recon', 'summer_recon', 'winter_global_pred', 'summer_global_pred']
    counts = aligned.dropna(subset=values).groupby('RGIId').year.nunique()
    fixed_ids = counts.index[counts.eq(47)]
    aligned = aligned[aligned.RGIId.isin(fixed_ids) & aligned.O1Region.ne(19)].copy()
    n_fixed = aligned.RGIId.nunique()

    model_dir = root / 'models/annual_seasonal'
    payloads = {}
    for season in SEASONS:
        with (model_dir / f'preprocessed_{season}_data.pkl').open('rb') as handle:
            payloads[season] = dill.load(handle)
    regions = supported_regions(payloads['winter']) & supported_regions(payloads['summer'])
    aligned = aligned[aligned.O1Region.isin(regions)].copy()
    cohort_counts = aligned.groupby('O1Region').RGIId.nunique().to_dict()
    print(f'Complete annual and seasonal series for {n_fixed} glaciers; Figure 9 cohort by region: {cohort_counts}.')

    meta = pd.read_csv(root / 'data/interim/wgms_targets/wgms_global_targets_with_rgi_1979_2025.csv')
    meta = meta.sort_values('RGIId').drop_duplicates('RGIId')
    era = pd.read_csv(root / 'data/raw/era5land_gee/wgms_global_era5land_1979_2025.csv',
                      usecols=['RGIId', 'year', 'month'] + STATE + FLOW)
    era = era.drop_duplicates()
    if era.duplicated(['RGIId', 'year', 'month']).any():
        raise ValueError('Monthly forcing contains conflicting entries; rerun 01_Dataset.ipynb.')
    checks, summary_frames, glacier_frames = {}, [], []
    for season, product in zip(SEASONS, (winter, summer)):
        print(f'Rebuilding {season} window members', flush=True)
        features = list(payloads[season]['feature_columns'])
        with (model_dir / f'xgboost_{season}_global.pkl').open('rb') as handle:
            model = dill.load(handle)
        records = rebuild_records(product, meta, era).dropna(subset=features).reset_index(drop=True)
        records['prediction'] = model.predict(records[features])
        recovered = records.groupby(keys).prediction.mean().rename('recovered').reset_index().merge(
            product[keys + [f'{season}_global_pred']], on=keys, validate='one_to_one',
        )
        err = float(np.max(np.abs(recovered.recovered - recovered[f'{season}_global_pred'])))
        if len(recovered) != len(product) or not np.isfinite(err) or err > 2e-5:
            raise AssertionError(f'{season} product mismatch: n={len(recovered)}/{len(product)}, max={err}')
        print(f'{season}: verified {len(recovered)} predictions, max error {err:.2g}',flush=True)
        records=records[records.RGIId.isin(aligned.RGIId.unique())].reset_index(drop=True)
        contributions=model.get_booster().predict(xgb.DMatrix(records[features],feature_names=features),pred_contribs=True)
        additive_error=float(np.max(np.abs(contributions.sum(axis=1)-records.prediction)))
        if additive_error>2e-5: raise AssertionError(f'SHAP additivity {additive_error}')
        for group, group_features in GROUPS.items():
            records[group]=contributions[:,[features.index(f) for f in group_features]].sum(axis=1)
        records['Bias']=contributions[:,-1]
        terms=list(GROUPS)+['Bias','prediction']
        per_year=records.groupby(keys)[terms].mean().reset_index().merge(aligned[keys+['O1Region','Area',f'{season}_recon']],on=keys,validate='one_to_one')
        per_year['Observation adjustment']=per_year[f'{season}_recon']-per_year.prediction
        terms+=['Observation adjustment',f'{season}_recon']
        per_year.to_csv(out / f'regional_change_{season}_glacier_year.csv', index=False)
        glacier=period_delta(per_year,terms)
        glacier.insert(0,'season',season)
        glacier_frames.append(glacier)
        summary=regional_summary(glacier,terms)
        summary.insert(0,'season',season)
        summary_frames.append(summary)
        checks[season]=dict(n_product_predictions=len(recovered),max_recovery_error=err,
                            max_shap_additivity_error=additive_error,n_fixed_glaciers=n_fixed,
                            n_figure_glaciers=records.RGIId.nunique(),
                            bias_range=float(np.ptp(contributions[:,-1])))
    pd.concat(summary_frames, ignore_index=True).to_csv(out / 'regional_change_attribution.csv', index=False)
    pd.concat(glacier_frames, ignore_index=True).to_csv(out / 'regional_change_attribution_glacier.csv', index=False)
    checks['cohort'] = {'complete_series': int(n_fixed), 'glaciers_by_region': {int(k): int(v) for k, v in cohort_counts.items()}}
    checks['method']={'early':[1979,1999],'late':[2000,2025],'bootstrap':'500 glacier resamples, fixed fitted models; 2.5/97.5 percentiles describe sample sensitivity',
                      'weights':'constant glacier areas; equal years within each period; equal valid window members',
                      'geometry':'signed allocation includes climate interactions, not changing geometry',
                      'scope':'same study glaciers with annual, winter and summer products for all 47 years; no model retraining'}
    (out / 'regional_change_attribution_metadata.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(environ.get('ML4GM_ROOT', DEFAULT_ROOT)),
                        help='Directory containing the inputs and generated products.')
    run(parser.parse_args().root.expanduser().resolve())
