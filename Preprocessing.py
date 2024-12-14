import pandas as pd
import ast
recipes1 = pd.read_csv('data/PP_recipes.csv')
recipes2 = pd.read_csv('data/RAW_recipes.csv')
recipes1_columns = ['id', 'ingredient_ids']
recipes2_columns = ['id', 'sodium', 'calories', 'fat', 'sugar', 'protein', 'saturated_fat']
recipes2['sodium'] = recipes2['nutrition'].apply(lambda x: ast.literal_eval(x)[3])
recipes2['calories'] = recipes2['nutrition'].apply(lambda x: ast.literal_eval(x)[0])
recipes2['fat'] = recipes2['nutrition'].apply(lambda x: ast.literal_eval(x)[1])
recipes2['sugar'] = recipes2['nutrition'].apply(lambda x: ast.literal_eval(x)[2])
recipes2['protein'] = recipes2['nutrition'].apply(lambda x: ast.literal_eval(x)[4])
recipes2['saturated_fat'] = recipes2['nutrition'].apply(lambda x: ast.literal_eval(x)[5])

recipes1 = recipes1[recipes1_columns]
recipes2 = recipes2[recipes2_columns]
df_recipes = pd.merge(recipes1,recipes2, on="id", how="inner")
df_recipes.to_csv('recipes.csv')