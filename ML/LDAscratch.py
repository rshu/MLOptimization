import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

feature_dict = {i: label for i, label in zip(
    range(4),
    ('sepal length in cm',
     'sepal width in cm',
     'petal length in cm',
     'petal width in cm',))}

df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
)
df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True)  # to drop the empty line at file-end

# print(df.tail())

X = df[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
# print(X)
y = df['class label'].values
# print(y)

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}

# Step 1: Computing the d-dimensional mean vectors
np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1, 4):
    mean_vectors.append(np.mean(X[y == cl], axis=0))
    print('Mean Vector class %s: %s\n' % (cl, mean_vectors[cl - 1]))

# Step 2: Computing the Scatter Matrices
# Within-class scatter matrix
S_W = np.zeros((4, 4))
for cl, mv in zip(range(1, 4), mean_vectors):
    class_sc_mat = np.zeros((4, 4))  # scatter matrix for every class
    for row in X[y == cl]:
        row, mv = row.reshape(4, 1), mv.reshape(4, 1)  # make column vectors
        class_sc_mat += (row - mv).dot((row - mv).T)
    S_W += class_sc_mat  # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)
print("")

# Between-class scatter matrix
overall_mean = np.mean(X, axis=0)

S_B = np.zeros((4, 4))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(4, 1)  # make column vector
    overall_mean = overall_mean.reshape(4, 1)  # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)
print("")

# Step 3: Solving the generalized eigenvalue problem for the matrix
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(4, 1)
    print('\nEigenvector {}: \n{}'.format(i + 1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i + 1, eig_vals[i].real))

# Checking the eigenvector-eigenvalue calculation
for i in range(len(eig_vals)):
    eigv = eig_vecs[:, i].reshape(4, 1)
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                         eig_vals[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
print('ok\n')

# Step 4: Selecting linear discriminants for the new feature subspace
# Sorting the eigenvectors by decreasing eigenvalues
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

# Choosing k eigenvectors with the largest eigenvalues
W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
print('Matrix W:\n', W.real)

# Step 5: Transforming the samples onto the new subspace
X_lda = X.dot(W)
assert X_lda.shape == (150, 2), "The matrix is not 150x2 dimensional."


def plot_step_lda():
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=X_lda[:, 0].real[y == label],
                    y=X_lda[:, 1].real[y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()


plot_step_lda()

sklearn_pca = sklearnPCA(n_components=2)
X_pca = sklearn_pca.fit_transform(X)


def plot_pca():
    ax = plt.subplot(111)

    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=X_pca[:, 0][y == label],
                    y=X_pca[:, 1][y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('PCA: Iris projection onto the first 2 principal components')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout
    plt.grid()
    plt.show()


plot_pca()

# LDA via scikit-learn
# LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)


def plot_scikit_lda(X, title):
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=X[:, 0][y == label],
                    y=X[:, 1][y == label] * -1,  # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()


plot_scikit_lda(X_lda_sklearn, title='Default LDA via scikit-learn')
