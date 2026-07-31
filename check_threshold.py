import joblib

PATH = "models/credit_default_model.joblib"

package = joblib.load(PATH)
print("Current saved threshold:", package["threshold"])


package["threshold"] = 0.424
joblib.dump(package, PATH)
print("Updated threshold to:", package["threshold"])
