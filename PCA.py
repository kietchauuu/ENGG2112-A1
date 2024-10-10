from sklearn.decomposition import PCA

# Standardize your data before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to, say, 10 components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)


from sklearn.feature_selection import SelectKBest, f_classif

# Select top k features using ANOVA F-test
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
