WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT('POINT(1 1)', 0)])
)
SELECT
  *,
  ST_REGIONSTATS(
    `bfcol_0`,
    'ee://some/raster/uri',
    band => 'band1',
    include => 'some equation',
    options => JSON '{"scale": 100}'
  ).`min` AS `min`,
  ST_REGIONSTATS(
    `bfcol_0`,
    'ee://some/raster/uri',
    band => 'band1',
    include => 'some equation',
    options => JSON '{"scale": 100}'
  ).`max` AS `max`,
  ST_REGIONSTATS(
    `bfcol_0`,
    'ee://some/raster/uri',
    band => 'band1',
    include => 'some equation',
    options => JSON '{"scale": 100}'
  ).`sum` AS `sum`,
  ST_REGIONSTATS(
    `bfcol_0`,
    'ee://some/raster/uri',
    band => 'band1',
    include => 'some equation',
    options => JSON '{"scale": 100}'
  ).`count` AS `count`,
  ST_REGIONSTATS(
    `bfcol_0`,
    'ee://some/raster/uri',
    band => 'band1',
    include => 'some equation',
    options => JSON '{"scale": 100}'
  ).`mean` AS `mean`,
  ST_REGIONSTATS(
    `bfcol_0`,
    'ee://some/raster/uri',
    band => 'band1',
    include => 'some equation',
    options => JSON '{"scale": 100}'
  ).`area` AS `area`
FROM `bfcte_0`