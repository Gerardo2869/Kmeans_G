Análisis de Clustering en Dataset de Pokémon
Descripción del Proyecto
Este proyecto aplica técnicas de clustering para segmentar un conjunto de datos de Pokémon según sus atributos de batalla. Utilizando el algoritmo de KMeans, buscamos identificar grupos en función de características clave como el ataque y la defensa. El análisis incluye la selección de atributos numéricos, la imputación de datos faltantes, y la visualización del número óptimo de clusters mediante el método del codo.

Archivos Incluidos
Dataset: pokemon.csv (contiene información de Pokémon, incluyendo su nombre, tipo, estadísticas base, y más).
Script de Análisis: Kmeans_G.ipynb (un notebook de Google Colab con todos los pasos de análisis, visualización y clustering).
README.md: Este archivo, con el resumen del flujo de trabajo y las instrucciones para ejecutar el proyecto.
Requisitos
Para ejecutar este proyecto, necesitarás las siguientes bibliotecas de Python:

pandas
numpy
matplotlib
seaborn
scikit-learn
yellowbrick
Flujo de Trabajo
Montaje de Google Drive: Se monta Google Drive para acceder al dataset almacenado.

python
from google.colab import drive
drive.mount("/content/drive/")
Carga y Exploración de Datos: Se carga el dataset y se visualizan los primeros datos y las columnas únicas de tipo de Pokémon.

python
import pandas as pd
basic1 = pd.read_csv('Ubicación del dataset en drive')
basic1.head()
Visualización de Clusters: Se define la función visualize_clusters para visualizar los clusters en función de los atributos base_attack y base_defense de los Pokémon.

Clustering con KMeans: Se utiliza el algoritmo KMeans para clasificar los datos en cuatro clusters. Como KMeans no acepta datos de texto ni valores NaN, se seleccionan sólo las columnas numéricas, y se realiza imputación con la media para completar los valores faltantes.

python
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Imputar valores NaN con la media
imputer = SimpleImputer(strategy='mean')
numeric_columns_imputed = imputer.fit_transform(basic_1_1.select_dtypes(include=[np.number]))

# Aplicar KMeans
kmeans_basic1 = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init='auto')
y_kmeans_basic1 = kmeans_basic1.fit_predict(numeric_columns_imputed)
Método del Codo: Para determinar el número óptimo de clusters, se utiliza el método del codo visualizando el WCSS (Within-Cluster Sum of Squares).

Método Manual:

python
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=300, init='k-means++', random_state=42, n_init='auto')
    kmeans.fit(numeric_columns_imputed)
    wcss.append(kmeans.inertia_)

# Graficar el WCSS
import matplotlib.pyplot as plt
plt.plot(range(1, 10), wcss)
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()
Utilizando Yellowbrick:

python
from yellowbrick.cluster import kelbow_visualizer
kelbow_visualizer(KMeans(random_state=4), numeric_columns_imputed, k=(2,10))
Resultados y Visualización Final: Se visualizan los clusters utilizando matplotlib para verificar las agrupaciones.

Conclusiones
El análisis permitió segmentar los Pokémon en cuatro clusters en función de sus atributos base de ataque y defensa. A través del método del codo, se determinó que un número adecuado de clusters era 4. Este tipo de análisis es útil para observar patrones de agrupación entre las diferentes especies y podría ampliarse con otros atributos o técnicas de clustering.
