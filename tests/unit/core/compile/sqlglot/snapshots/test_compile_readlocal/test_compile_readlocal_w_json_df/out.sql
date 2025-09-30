SELECT
`rowindex` AS `rowindex`,
`json_col` AS `json_col`
FROM
(SELECT
  `t0`.`level_0` AS `rowindex`,
  `t0`.`column_0` AS `json_col`,
  `t0`.`bfuid_col_902` AS `bfuid_col_903`
FROM (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`level_0` INT64, `column_0` JSON, `bfuid_col_902` INT64>>[STRUCT(0, PARSE_JSON('null'), 0), STRUCT(1, PARSE_JSON('true'), 1), STRUCT(2, PARSE_JSON('100'), 2), STRUCT(3, PARSE_JSON('0.98'), 3), STRUCT(4, PARSE_JSON('"a string"'), 4), STRUCT(5, PARSE_JSON('[]'), 5), STRUCT(6, PARSE_JSON('[1,2,3]'), 6), STRUCT(7, PARSE_JSON('[{"a":1},{"a":2},{"a":null},{}]'), 7), STRUCT(8, PARSE_JSON('"100"'), 8), STRUCT(9, PARSE_JSON('{"date":"2024-07-16"}'), 9), STRUCT(10, PARSE_JSON('{"int_value":2,"null_filed":null}'), 10), STRUCT(11, PARSE_JSON('{"list_data":[10,20,30]}'), 11)]) AS `level_0`
) AS `t0`)
ORDER BY `bfuid_col_903` ASC NULLS LAST