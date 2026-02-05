WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT('POINT(1 1)', 0)])
)
SELECT
  *,
  ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`min` AS `min`,
  ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`max` AS `max`,
  ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`sum` AS `sum`,
  ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`count` AS `count`,
  ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`mean` AS `mean`,
  ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`area` AS `area`
FROM `bfcte_0`