# Menentukan criteria dalam menghitung Information Gain

import numpy as np

# CLASSIFICATION

def Gini(y):
    """
    Menghitung Gini impurity dari array target (y).

    Args:
        y (array-like): Array target yang berisi label kelas.

    Returns:
        float: Gini impurity dari node.

    """
    # Melakukan ekstraksi pada class

    K, counts = np.unique(y, return_counts =True)
    counts_unique = dict(zip(K, counts))
    N_m = len(y)

    # Melakukan kalkulasi proporsi dari observasi class k
    p_m = {}
    for k in K:
        p_m[k] = counts_unique[k] / N_m

    # Menghitung impurity dari node
    impurity_node = 0
    for k in K:
        impurity_node += p_m[k] * (1-p_m[k])
    
    return impurity_node

def Log_Loss(y):
    """
    Menghitung Log Loss impurity dari array target (y).

    Args:
        y (array-like): Array target yang berisi label kelas.

    Returns:
        float: Log Loss impurity dari node.

    """
    # Melakukan ekstraksi pada class

    K, counts = np.unique(y, return_counts =True)
    counts_unique = dict(zip(K, counts))
    N_m = len(y)

    # Melakukan kalkulasi proporsi dari observasi class k
    p_m = {}
    for k in K:
        p_m[k] = counts_unique[k] / N_m

    # Menghitung majority class pada node m
    max_ind = np.argmax(counts)
    max_class = K[max_ind] 

    # Menghitung impurity dari node
    impurity_node = 1 - p_m[max_class]
    
    return impurity_node

def Entropy(y):
    """
    Menghitung Entropy impurity dari array target (y).

    Args:
        y (array-like): Array target yang berisi label kelas.

    Returns:
        float: Entropy impurity dari node.

    """
    # Melakukan ekstraksi pada class

    K, counts = np.unique(y, return_counts =True)
    counts_unique = dict(zip(K, counts))
    N_m = len(y)

    # Melakukan kalkulasi proporsi dari observasi class k
    p_m = {}
    for k in K:
        p_m[k] = counts_unique[k] / N_m

    # Menghitung impurity dari node
    impurity_node = 0
    for k in K:
        impurity_node += p_m[k] * np.log(p_m[k])

    return impurity_node

# REGRESSION
def MSE(y):
    """
    Menghitung Mean Squared Error (MSE) dari array target (y) untuk masalah regresi.

    Args:
        y (array-like): Array target yang berisi nilai-nilai target.

    Returns:
        float: MSE dari node.

    """
    # Menghitung jumlah data y
    n = len(y)

    # Mengurangi nilai y dengan rata-rata dari y 
    r = y - np.mean(y)

    # Kuadratkan nilai r
    r = r ** 2

    # Menjumlahkan nilai dari y
    r = np.sum(r)
        
    return r / n 

def MAE(y):
    """
    Menghitung Mean Absolute Error (MAE) dari array target (y) untuk masalah regresi.

    Args:
        y (array-like): Array target yang berisi nilai-nilai target.

    Returns:
        float: MAE dari node.

    """
    # Menghitung jumlah data y
    n = len(y)

    # Mengurangi nilai y dengan rata-rata dari y dan absolutkan
    r = abs(y - np.mean(y))

    # Menjumlahkan nilai dari y
    r = np.sum(r)
        
    return r / n 