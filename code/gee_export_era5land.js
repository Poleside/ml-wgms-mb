// Earth Engine Code Editor script for exporting monthly ERA5-Land values
// at WGMS-RGI glacier centroids.
//
// How to use:
// 1. Upload the centroid CSV as an Earth Engine table asset with at least:
//    RGIId, CenLon, CenLat, O1Region, O2Region
// 2. Replace POINT_ASSET with the uploaded asset id.
// 3. Optionally set REGION_FILTER / YEAR_MIN / YEAR_MAX. For a separate
//    1978 boundary export, set both year bounds to 1978 and leave
//    REGION_FILTER unset. Use the same point asset as for 1979–2025.
// 4. Run this script in the Earth Engine Code Editor: https://code.earthengine.google.com/
// 5. Start the table export task from the Tasks panel.

var POINT_ASSET = 'projects/YOUR_PROJECT/assets/wgms_global_glacier_points_for_gee';
var DRIVE_FOLDER = 'WGMS_MB_ERA5LAND';  // Download into data/raw/era5land_gee/exports/.
var YEAR_MIN = 1979;
var YEAR_MAX = 2025;
var REGION_FILTER = null;  // e.g. [13, 14, 15] or null
var EXPORT_PREFIX = 'wgms_era5land_monthly_' + YEAR_MIN + '_' + YEAR_MAX;
if (REGION_FILTER !== null) {
  EXPORT_PREFIX += '_rgi_' + REGION_FILTER.join('_');
}

var ERA5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR');

var BAND_MAP = {
  t2m: 'temperature_2m',
  rsn: 'snow_density',
  sde: 'snow_depth_water_equivalent',
  sf: 'snowfall_sum',
  smlt: 'snowmelt_sum',
  tsn: 'temperature_of_snow_layer',
  fal: 'forecast_albedo',
  slhf: 'surface_latent_heat_flux_sum',
  ssr: 'surface_net_solar_radiation_sum',
  strd: 'surface_thermal_radiation_downwards_sum',
  sshf: 'surface_sensible_heat_flux_sum',
  ssrd: 'surface_solar_radiation_downwards_sum',
  sp: 'surface_pressure',
  tp: 'total_precipitation_sum'
};

var renameBands = Object.keys(BAND_MAP);
var sourceBands = renameBands.map(function(name) { return BAND_MAP[name]; });

function makePointFeature(f) {
  var lon = ee.Number.parse(f.get('CenLon'));
  var lat = ee.Number.parse(f.get('CenLat'));
  return ee.Feature(
    ee.Geometry.Point([lon, lat]),
    {
      RGIId: f.get('RGIId'),
      O1Region: f.get('O1Region'),
      O2Region: f.get('O2Region')
    }
  );
}

var rawPoints = ee.FeatureCollection(POINT_ASSET);
var pointFc = ee.FeatureCollection(rawPoints.map(makePointFeature));

if (REGION_FILTER !== null) {
  pointFc = pointFc.filter(ee.Filter.inList('O1Region', REGION_FILTER));
}

print('Point asset preview', pointFc.limit(5));
print('Point count', pointFc.size());
Map.centerObject(pointFc, 3);
Map.addLayer(pointFc, {color: 'red'}, 'WGMS glacier centroids');

var years = ee.List.sequence(YEAR_MIN, YEAR_MAX);
var months = ee.List.sequence(1, 12);

function sampleMonth(year, month) {
  year = ee.Number(year);
  month = ee.Number(month);
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');
  var image = ee.Image(ERA5.filterDate(start, end).first())
    .select(sourceBands, renameBands);

  var samples = image.sampleRegions({
    collection: pointFc,
    scale: 11132,
    geometries: false
  });

  return samples.map(function(f) {
    return f.set({
      year: year,
      month: month
    });
  });
}

var monthlyCollections = years.map(function(year) {
  return months.map(function(month) {
    return sampleMonth(year, month);
  });
}).flatten();

var exportFc = ee.FeatureCollection(monthlyCollections).flatten();

print('Export preview', exportFc.limit(5));
print('Estimated row count', exportFc.size());

Export.table.toDrive({
  collection: exportFc,
  description: EXPORT_PREFIX,
  folder: DRIVE_FOLDER,
  fileNamePrefix: EXPORT_PREFIX,
  fileFormat: 'CSV'
});
