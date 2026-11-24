// Jalankan di https://code.earthengine.google.com  (login akun Google, daftar Earth Engine dulu - gratis untuk riset/non-komersial)
// Hasil: file chirps_sutojayan.csv di Google Drive -> simpan ke folder riset_v2/data/
var pts = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([112.248497, -8.176364]), {st: 'Judeg'}),
  ee.Feature(ee.Geometry.Point([112.229789, -8.184394]), {st: 'Bacem'})
]);

var ic = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2015-10-01', '2026-03-01');   // mulai 2015-10 sebagai 'pemanasan' untuk fitur lag; batas akhir tidak termasuk

var fc = ic.map(function (img) {
  return img.reduceRegions({collection: pts, reducer: ee.Reducer.mean(), scale: 5566})
    .map(function (f) { return f.set('date', img.date().format('YYYY-MM-dd')); });
}).flatten();

Export.table.toDrive({
  collection: fc,
  description: 'chirps_sutojayan',
  fileFormat: 'CSV',
  selectors: ['date', 'st', 'mean']
});
// Setelah klik Run, buka tab "Tasks" (kanan atas) -> klik RUN pada task 'chirps_sutojayan'.
