import shap
from sklearn.ensemble import RandomForestClassifier
import pandas

X = pandas.read_csv('./data/imputed_wearable.csv')
y = pandas.read_csv('./data/wearable.csv')

y = y['therm_pref']

model = RandomForestClassifier()
model.fit(X, y)
shap_values = shap.TreeExplainer(model).shap_values(X)
shap.summary_plot(shap_values, X, plot_type='bar')
