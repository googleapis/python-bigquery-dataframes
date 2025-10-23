WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` TIMESTAMP, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS TIMESTAMP), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    TIMESTAMP_DIFF(
      `bfcol_0`,
      LAG(`bfcol_0`, 1) OVER (ORDER BY `bfcol_0` ASC NULLS LAST, `bfcol_1` ASC NULLS LAST),
      MICROSECOND
    ) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `diff_time`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST