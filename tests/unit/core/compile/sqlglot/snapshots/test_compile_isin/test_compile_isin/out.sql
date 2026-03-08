WITH `bfcte_0` AS (
  SELECT
    `int64_too`
  FROM (
    SELECT
      `int64_too`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
  GROUP BY
    `int64_too`
), `bfcte_1` AS (
  SELECT
    *,
    STRUCT(COALESCE(`bfcol_4`, 0) AS `bfpart1_0`, COALESCE(`bfcol_4`, 1) AS `bfpart2_0`) IN (
      (
        SELECT
          STRUCT(COALESCE(`bfcol_0`, 0) AS `bfpart1_0`, COALESCE(`bfcol_0`, 1) AS `bfpart2_0`)
        FROM (
          SELECT
            `int64_too` AS `bfcol_0`
          FROM `bfcte_0`
        )
      )
    ) AS `bfcol_5`
  FROM (
    SELECT
      `rowindex` AS `bfcol_3`,
      `int64_col` AS `bfcol_4`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)
SELECT
  `bfcol_3` AS `rowindex`,
  `bfcol_5` AS `int64_col`
FROM `bfcte_1`