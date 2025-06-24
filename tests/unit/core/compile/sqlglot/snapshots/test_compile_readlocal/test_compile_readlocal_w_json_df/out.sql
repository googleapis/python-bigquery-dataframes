WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` STRING, `bfcol_2` INT64>>[STRUCT(0, 'null', 0), STRUCT(1, 'true', 1), STRUCT(2, '100', 2), STRUCT(3, '0.98', 3), STRUCT(4, '"a string"', 4), STRUCT(5, '[]', 5), STRUCT(6, '[1, 2, 3]', 6), STRUCT(7, '[{"a": 1}, {"a": 2}, {"a": null}, {}]', 7), STRUCT(8, '"100"', 8), STRUCT(9, '{"date": "2024-07-16"}', 9), STRUCT(10, '{"int_value": 2, "null_filed": null}', 10), STRUCT(11, '{"list_data": [10, 20, 30]}', 11)])
)
SELECT
  `bfcol_0` AS `rowindex`,
  `bfcol_1` AS `json_col`
FROM `bfcte_0`
ORDER BY
  `bfcol_2` ASC NULLS LAST