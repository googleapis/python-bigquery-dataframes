SELECT
  (
    SELECT
      ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`min`,
      ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`max`,
      ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`sum`,
      ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`count`,
      ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`mean`,
      ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`area`
    FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT('POINT(1 1)', 0)])
  )
  ORDER BY
    `bfcol_1` ASC NULLS LAST.*
FROM (
  SELECT
    ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`min`,
    ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`max`,
    ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`sum`,
    ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`count`,
    ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`mean`,
    ST_REGIONSTATS(`bfcol_0`, 'ee://some/raster/uri').`area`
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT('POINT(1 1)', 0)])
)
ORDER BY
  `bfcol_1` ASC NULLS LAST