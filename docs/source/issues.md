
1. clip vector error
- description
```
/usr/local/lib/python3.6/site-packages/geopandas/base.py:76: UserWarning: Cannot generate spatial index: Missing package `rtree`.
  warn("Cannot generate spatial index: Missing package `rtree`.")
Traceback (most recent call last):
  File "script.py", line 17, in <module>
    res_union = gpd.overlay(df1, df2, how='union')
  File "/usr/local/lib/python3.6/site-packages/geopandas/tools/overlay.py", line 371, in overlay
    result = _overlay_union(df1, df2)
  File "/usr/local/lib/python3.6/site-packages/geopandas/tools/overlay.py", line 298, in _overlay_union
    dfinter = _overlay_intersection(df1, df2)
  File "/usr/local/lib/python3.6/site-packages/geopandas/tools/overlay.py", line 212, in _overlay_intersection
    sidx = bbox.apply(lambda x: list(spatial_index.intersection(x)))
  File "/usr/local/lib/python3.6/site-packages/pandas/core/series.py", line 3194, in apply
    mapped = lib.map_infer(values, f, convert=convert_dtype)
  File "pandas/_libs/src/inference.pyx", line 1472, in pandas._libs.lib.map_infer
  File "/usr/local/lib/python3.6/site-packages/geopandas/tools/overlay.py", line 212, in <lambda>
    sidx = bbox.apply(lambda x: list(spatial_index.intersection(x)))
AttributeError: 'NoneType' object has no attribute 'intersection'
```

- solution: you should install the rtree
for macos: brew install spatialindex && pip3 install rtree