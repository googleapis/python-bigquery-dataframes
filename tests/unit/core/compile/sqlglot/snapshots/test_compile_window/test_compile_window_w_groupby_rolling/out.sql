WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` INT64, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_2` AS `bfcol_8`,
    `bfcol_0` AS `bfcol_9`,
    `bfcol_1` AS `bfcol_10`,
    `bfcol_0` AS `bfcol_11`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_1`
  WHERE
    NOT `bfcol_11` IS NULL
), `bfcte_3` AS (
  SELECT
    *,
    CASE
      WHEN SUM(CAST(NOT `bfcol_9` IS NULL AS INT64)) OVER (
        PARTITION BY `bfcol_11`
        ORDER BY `bfcol_11` ASC NULLS LAST, `bfcol_2` ASC NULLS LAST, `bfcol_3` ASC NULLS LAST
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
      ) < 3
      THEN NULL
      ELSE COALESCE(
        SUM(CAST(`bfcol_9` AS INT64)) OVER (
          PARTITION BY `bfcol_11`
          ORDER BY `bfcol_11` ASC NULLS LAST, `bfcol_2` ASC NULLS LAST, `bfcol_3` ASC NULLS LAST
          ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ),
        0
      )
    END AS `bfcol_18`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_3`
  WHERE
    NOT `bfcol_11` IS NULL
), `bfcte_5` AS (
  SELECT
    *,
    CASE
      WHEN SUM(CAST(NOT `bfcol_10` IS NULL AS INT64)) OVER (
        PARTITION BY `bfcol_11`
        ORDER BY `bfcol_11` ASC NULLS LAST, `bfcol_2` ASC NULLS LAST, `bfcol_3` ASC NULLS LAST
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
      ) < 3
      THEN NULL
      ELSE COALESCE(
        SUM(`bfcol_10`) OVER (
          PARTITION BY `bfcol_11`
          ORDER BY `bfcol_11` ASC NULLS LAST, `bfcol_2` ASC NULLS LAST, `bfcol_3` ASC NULLS LAST
          ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ),
        0
      )
    END AS `bfcol_25`
  FROM `bfcte_4`
)
SELECT
  `bfcol_11` AS `bool_col`,
  `bfcol_8` AS `rowindex`,
  `bfcol_18` AS `bool_col_1`,
  `bfcol_25` AS `int64_col`
FROM `bfcte_5`
ORDER BY
  `bfcol_11` ASC NULLS LAST,
  `bfcol_2` ASC NULLS LAST,
  `bfcol_3` ASC NULLS LAST