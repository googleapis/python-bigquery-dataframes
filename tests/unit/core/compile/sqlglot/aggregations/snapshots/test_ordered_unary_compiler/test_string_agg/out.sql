SELECT
  `bfcol_1` AS `string_col`
FROM (
  SELECT
    COALESCE(
      STRING_AGG(`string_col`, ','
      ORDER BY
        `string_col` IS NULL ASC,
        `string_col` ASC),
      ''
    ) AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)