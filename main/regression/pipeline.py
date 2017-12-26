from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from .CombinedAttributesAdder import CombinedAttributesAdder

num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
