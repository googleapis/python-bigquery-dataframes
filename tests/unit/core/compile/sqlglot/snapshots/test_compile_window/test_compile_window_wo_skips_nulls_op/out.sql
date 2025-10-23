WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    CASE
      WHEN COUNT(CAST(NOT `bfcol_0` IS NULL AS INT64)) OVER (
        ORDER BY `bfcol_1` ASC NULLS LAST, `bfcol_2` ASC NULLS LAST
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
      ) < 5
      THEN NULL
      ELSE COUNT(`bfcol_0`) OVER (
        ORDER BY `bfcol_1` ASC NULLS LAST, `bfcol_2` ASC NULLS LAST
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
      )
    END AS `bfcol_6`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `rowindex`,
  `bfcol_6` AS `int64_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST,
  `bfcol_2` ASC NULLS LAST