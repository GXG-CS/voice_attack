# Test the BaggingClassifier in isolation
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)
    print("BaggingClassifier instantiated successfully.")
except TypeError as e:
    print(f"TypeError encountered: {e}")
