# Data layout

Place Sentinel-1 patches in `data/sar/`, one folder per class:

```
data/sar/
  Forest/   scene01__0001.npy  scene01__0002.npy
  Urban/    scene02__0001.npy
  Water/    scene03__0001.npy
```

- Each patch has shape `(2, H, W)` with channels `[VV, VH]`, as `.npy` or a 2-band GeoTIFF (needs rasterio).
- Values are linear backscatter. Pass `--input-db` if they are in dB.
- Name files `<scene>__<index>` so `--split group` keeps each scene in one part.
