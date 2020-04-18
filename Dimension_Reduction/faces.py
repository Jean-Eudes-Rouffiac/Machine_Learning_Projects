from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# Chargement du dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70)

# Recupération de la taille des images pour plot
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data  # Images vectorisées
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Nombre d'images : %d" % n_samples)
print("Chaque image fait {} par {} pixels".format(h, w))
print("Nombre de variables : %d" % n_features)
print("Nombre de classes : %d" % n_classes)
print(target_names)

image_titles = ["Image %d" % i for i in range(lfw_people.images.shape[0])]
plot_gallery(lfw_people.images, image_titles, h, w)
plt.show()

# Séparation train/test
# NE PAS CHANGER
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
