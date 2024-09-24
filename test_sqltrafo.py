from bigframes.ml import compose

trafo1 = compose.SQLScalarColumnTransformer("{0}+1")
trafo2 = compose.SQLScalarColumnTransformer("{0}-1")
print(trafo1)
print(trafo1)
print(trafo1==trafo2)
