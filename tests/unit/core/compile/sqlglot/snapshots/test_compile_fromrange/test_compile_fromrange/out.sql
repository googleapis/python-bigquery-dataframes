SELECT `bigframes_unnamed_index`, `int64_col`, `int64_too` FROM (WITH `t3` AS (
  SELECT
    `t1`.`column_0` AS `bfuid_col_1571`
  FROM (
    SELECT
      *
    FROM UNNEST(ARRAY<STRUCT<`column_0` DATETIME>>[STRUCT(DATETIME('2021-01-01T13:00:00')), STRUCT(DATETIME('2021-01-01T13:00:01')), STRUCT(DATETIME('2021-01-01T13:00:02')), STRUCT(DATETIME('2021-01-01T13:00:03')), STRUCT(DATETIME('2021-01-01T13:00:04')), STRUCT(DATETIME('2021-01-01T13:00:05')), STRUCT(DATETIME('2021-01-01T13:00:06')), STRUCT(DATETIME('2021-01-01T13:00:07')), STRUCT(DATETIME('2021-01-01T13:00:08')), STRUCT(DATETIME('2021-01-01T13:00:09')), STRUCT(DATETIME('2021-01-01T13:00:10')), STRUCT(DATETIME('2021-01-01T13:00:11')), STRUCT(DATETIME('2021-01-01T13:00:12')), STRUCT(DATETIME('2021-01-01T13:00:13')), STRUCT(DATETIME('2021-01-01T13:00:14')), STRUCT(DATETIME('2021-01-01T13:00:15')), STRUCT(DATETIME('2021-01-01T13:00:16')), STRUCT(DATETIME('2021-01-01T13:00:17')), STRUCT(DATETIME('2021-01-01T13:00:18')), STRUCT(DATETIME('2021-01-01T13:00:19')), STRUCT(DATETIME('2021-01-01T13:00:20')), STRUCT(DATETIME('2021-01-01T13:00:21')), STRUCT(DATETIME('2021-01-01T13:00:22')), STRUCT(DATETIME('2021-01-01T13:00:23')), STRUCT(DATETIME('2021-01-01T13:00:24')), STRUCT(DATETIME('2021-01-01T13:00:25')), STRUCT(DATETIME('2021-01-01T13:00:26')), STRUCT(DATETIME('2021-01-01T13:00:27')), STRUCT(DATETIME('2021-01-01T13:00:28')), STRUCT(DATETIME('2021-01-01T13:00:29'))]) AS `column_0`
  ) AS `t1`
), `t11` AS (
  SELECT
    CAST(FLOOR(
      ieee_divide(
        UNIX_MICROS(CAST(`t6`.`bfuid_col_1571` AS TIMESTAMP)) - UNIX_MICROS(CAST(CAST(`t9`.`bfuid_col_1572` AS DATE) AS TIMESTAMP)),
        7000000.0
      )
    ) AS INT64) AS `bfuid_col_1573`
  FROM `t3` AS `t6`
  CROSS JOIN (
    SELECT
      *
    FROM (
      SELECT
        MIN(`t5`.`bfuid_col_1571`) AS `bfuid_col_1572`
      FROM `t3` AS `t5`
    ) AS `t7`
  ) AS `t9`
)
SELECT
  CAST(CAST(timestamp_micros(
    CAST(trunc(
      (
        `t27`.`labels` * 7000000.0
      ) + UNIX_MICROS(CAST(CAST(`t27`.`bfuid_col_1572` AS DATE) AS TIMESTAMP))
    ) AS INT64)
  ) AS TIMESTAMP) AS DATETIME) AS `bigframes_unnamed_index`,
  `t14`.`column_1` AS `int64_col`,
  `t14`.`column_2` AS `int64_too`,
  `t27`.`labels` AS `bfuid_col_1578`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t24`.`labels`,
      `t9`.`bfuid_col_1572`
    FROM (
      SELECT
        *
      FROM (
        SELECT
          `t21`.*
          REPLACE (`ibis_table_unnest_column_fq25i2xrkzew3pldbtkhcneozu` AS `labels`)
        FROM (
          SELECT
            IF(
              NOT NULLIF(1, 0) IS NULL
              AND SIGN(1) = SIGN((
                `t20`.`bfuid_col_1575` + 1
              ) - `t19`.`bfuid_col_1574`),
              ARRAY(
                SELECT
                  ibis_bq_arr_range_fqkrald5tbbe5m6vt2tohfy3fi
                FROM UNNEST(generate_array(`t19`.`bfuid_col_1574`, `t20`.`bfuid_col_1575` + 1, 1)) AS ibis_bq_arr_range_fqkrald5tbbe5m6vt2tohfy3fi
                WHERE
                  ibis_bq_arr_range_fqkrald5tbbe5m6vt2tohfy3fi <> (
                    `t20`.`bfuid_col_1575` + 1
                  )
              ),
              []
            ) AS `labels`
          FROM (
            SELECT
              *
            FROM (
              SELECT
                MIN(`t12`.`bfuid_col_1573`) AS `bfuid_col_1574`
              FROM `t11` AS `t12`
            ) AS `t15`
          ) AS `t19`
          CROSS JOIN (
            SELECT
              *
            FROM (
              SELECT
                MAX(`t12`.`bfuid_col_1573`) AS `bfuid_col_1575`
              FROM `t11` AS `t12`
            ) AS `t16`
          ) AS `t20`
        ) AS `t21`
        CROSS JOIN UNNEST(`t21`.`labels`) AS `ibis_table_unnest_column_fq25i2xrkzew3pldbtkhcneozu`
      ) AS `t22`
    ) AS `t24`
    CROSS JOIN (
      SELECT
        *
      FROM (
        SELECT
          MIN(`t5`.`bfuid_col_1571`) AS `bfuid_col_1572`
        FROM `t3` AS `t5`
      ) AS `t7`
    ) AS `t9`
  ) AS `t25`
) AS `t27`
LEFT OUTER JOIN (
  SELECT
    *
  FROM (
    SELECT
      `t4`.`column_1`,
      `t4`.`column_2`,
      CAST(FLOOR(
        ieee_divide(
          UNIX_MICROS(CAST(`t4`.`bfuid_col_1571` AS TIMESTAMP)) - UNIX_MICROS(CAST(CAST(`t9`.`bfuid_col_1572` AS DATE) AS TIMESTAMP)),
          7000000.0
        )
      ) AS INT64) AS `bfuid_col_1573`
    FROM (
      SELECT
        `t0`.`column_0` AS `bfuid_col_1571`,
        `t0`.`column_1`,
        `t0`.`column_2`
      FROM (
        SELECT
          *
        FROM UNNEST(ARRAY<STRUCT<`column_0` DATETIME, `column_1` INT64, `column_2` INT64>>[STRUCT(DATETIME('2021-01-01T13:00:00'), 0, 10), STRUCT(DATETIME('2021-01-01T13:00:01'), 1, 11), STRUCT(DATETIME('2021-01-01T13:00:02'), 2, 12), STRUCT(DATETIME('2021-01-01T13:00:03'), 3, 13), STRUCT(DATETIME('2021-01-01T13:00:04'), 4, 14), STRUCT(DATETIME('2021-01-01T13:00:05'), 5, 15), STRUCT(DATETIME('2021-01-01T13:00:06'), 6, 16), STRUCT(DATETIME('2021-01-01T13:00:07'), 7, 17), STRUCT(DATETIME('2021-01-01T13:00:08'), 8, 18), STRUCT(DATETIME('2021-01-01T13:00:09'), 9, 19), STRUCT(DATETIME('2021-01-01T13:00:10'), 10, 20), STRUCT(DATETIME('2021-01-01T13:00:11'), 11, 21), STRUCT(DATETIME('2021-01-01T13:00:12'), 12, 22), STRUCT(DATETIME('2021-01-01T13:00:13'), 13, 23), STRUCT(DATETIME('2021-01-01T13:00:14'), 14, 24), STRUCT(DATETIME('2021-01-01T13:00:15'), 15, 25), STRUCT(DATETIME('2021-01-01T13:00:16'), 16, 26), STRUCT(DATETIME('2021-01-01T13:00:17'), 17, 27), STRUCT(DATETIME('2021-01-01T13:00:18'), 18, 28), STRUCT(DATETIME('2021-01-01T13:00:19'), 19, 29), STRUCT(DATETIME('2021-01-01T13:00:20'), 20, 30), STRUCT(DATETIME('2021-01-01T13:00:21'), 21, 31), STRUCT(DATETIME('2021-01-01T13:00:22'), 22, 32), STRUCT(DATETIME('2021-01-01T13:00:23'), 23, 33), STRUCT(DATETIME('2021-01-01T13:00:24'), 24, 34), STRUCT(DATETIME('2021-01-01T13:00:25'), 25, 35), STRUCT(DATETIME('2021-01-01T13:00:26'), 26, 36), STRUCT(DATETIME('2021-01-01T13:00:27'), 27, 37), STRUCT(DATETIME('2021-01-01T13:00:28'), 28, 38), STRUCT(DATETIME('2021-01-01T13:00:29'), 29, 39)]) AS `column_0`
      ) AS `t0`
    ) AS `t4`
    CROSS JOIN (
      SELECT
        *
      FROM (
        SELECT
          MIN(`t5`.`bfuid_col_1571`) AS `bfuid_col_1572`
        FROM `t3` AS `t5`
      ) AS `t7`
    ) AS `t9`
  ) AS `t10`
) AS `t14`
  ON `t27`.`labels` = `t14`.`bfuid_col_1573`) AS `t`
ORDER BY `bfuid_col_1578` ASC NULLS LAST