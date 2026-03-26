SELECT
  CONCAT(
    CAST(FARM_FINGERPRINT(
      CONCAT(
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`rowindex` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`bool_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`bytes_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`date_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`datetime_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(st_astext(`t1`.`geography_col`), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`int64_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`int64_too` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`numeric_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`float64_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`rowindex` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`rowindex_2` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(`t1`.`string_col`, ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`time_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`timestamp_col` AS STRING), ''), '\\', '\\\\')),
        CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`duration_col` AS STRING), ''), '\\', '\\\\'))
      )
    ) AS STRING),
    CAST(FARM_FINGERPRINT(
      CONCAT(
        CONCAT(
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`rowindex` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`bool_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`bytes_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`date_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`datetime_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(st_astext(`t1`.`geography_col`), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`int64_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`int64_too` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`numeric_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`float64_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`rowindex` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`rowindex_2` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(`t1`.`string_col`, ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`time_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`timestamp_col` AS STRING), ''), '\\', '\\\\')),
          CONCAT('\\', REPLACE(COALESCE(CAST(`t1`.`duration_col` AS STRING), ''), '\\', '\\\\'))
        ),
        '_'
      )
    ) AS STRING),
    CAST(RAND() AS STRING)
  ) AS `row_key`
FROM (
  SELECT
    *
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`