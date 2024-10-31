from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Carregar o dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir o conjunto de dados para garantir que os modelos estejam comparáveis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar os modelos
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)

model_svm = SVC(probability=True)
model_svm.fit(X_train, y_train)

# Salvar os modelos
joblib.dump(model_rf, "modelo_rf.pkl")
joblib.dump(model_knn, "modelo_knn.pkl")
joblib.dump(model_svm, "modelo_svm.pkl")

