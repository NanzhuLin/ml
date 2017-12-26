from util import util
import matplotlib.pyplot as plt
import numpy as np
from .CombinedAttributesAdder import CombinedAttributesAdder

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

# util.fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
housing = util.load_data(HOUSING_PATH, "housing.csv")
housing.hist(bins=50, figsize=(20, 15))

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform()

