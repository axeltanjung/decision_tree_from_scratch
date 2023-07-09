import numpy as np

from . import _criteria

'============================================================================================================'
CRITERIA_CLF = {
    "gini": _criteria.Gini,
    "log_loss": _criteria.Log_Loss,
    "entropy": _criteria.Entropy
}

CRITERIA_REG = {
    "squared_error": _criteria.MSE,
    "absolute_error": _criteria.MAE
}

'============================================================================================================'

class Tree:
    """
    Fungsi yang digunakan untuk melakukan inisiasi Tree,

        - thresh        = Threshold untuk membagi region
        - fitur         = Fitur data untuk mendefinisikan root
        - val           = Nilai dari node
        - child_left    = Hasil split region children bagian kiri
        - child_right   = Hasil split region children bagian kanan
        - impurity      = Nilai impuritas dari data dalam region
        - leaf          = Apakah node tersebut leaf?
        - n_samples     = Total sample yang digunakan
    """
    def __init__(
        self,
        thresh=None,
        fitur=None,
        val=None,
        child_left=None,
        child_right=None,
        impurity=None,
        leaf=None,
        n_samples=None
    ):
        self.thresh = thresh
        self.fitur = fitur
        self.val = val
        self.child_left = child_left
        self.child_right = child_right
        self.impurity =  impurity
        self.leaf = leaf
        self.n_samples = n_samples

def _split(data):
    """
    Memisahkan data menjadi region-region menggunakan nilai threshold.

    Args:
        data (array-like): Data yang akan dipisahkan.

    Returns:
        numpy.ndarray: Array berisi nilai threshold yang digunakan untuk membagi data.
    """
    # Copy data untuk menjaga data integrity
    data = data.copy()

    # Mengambil nilai unik dari data
    val_unique = np.unique(data)

    # Mencari shape dari data
    n_shape = len(val_unique)

    # Mengurutkan data (agar dapat membagi region berdasarkan threshold)
    val_unique.sort()

    # Melakukan inisialisasi terhadap threshold
    thresh = np.zeros(n_shape-1)

    # Membuat split untuk mendapatkan region
    for i in range(n_shape-1):
        nilai_1 = val_unique[i]
        nilai_1 = val_unique[i+1]

        # Nilai threshold berada di tengah data nilai_1 dan nilai_2    
        thresh[i] = 0.5*(nilai_1 + nilai_1)
    
    return thresh

def _split_data(data, fitur, thresh):
    """
    Memisahkan data menjadi dua subset berdasarkan threshold pada fitur yang diberikan.

    Args:
        data (numpy.ndarray): Data yang akan dipisahkan.
        feature (int): Indeks fitur yang digunakan untuk pemisahan.
        threshold (float): Nilai threshold yang digunakan untuk membagi data.

    Returns:
        tuple: Tuple berisi dua subset data yang telah dipisahkan.

    """
    # Membagi data berdasarkan threshold
    data_thresh = data[:, fitur] <= thresh
    left_data = data[data_thresh]
    right_data = data[~data_thresh]

    return left_data, right_data

def _calculate_majority_vote(y):
    """
    Menghitung majority vote dari array target (y).

    Args:
        y (array-like): Array target yang berisi label kelas.

    Returns:
        Any: Label kelas hasil majority vote.

    """
    # Melakukan ekstraksi terhadap output
    vals, counts = np.unique(y, return_counts = True)

    # Menghitung majority vote
    max_ind = np.argmax(counts)
    y_pred = vals[max_ind]

    return y_pred

def _calculate_average_vote(y):
    """
    Menghitung rata-rata dari array target (y).

    Args:
        y (array-like): Array target yang berisi label kelas.

    Returns:
        float: Rata-rata label kelas.

    """
    # Menghitung rata-rata
    y_pred = np.mean(y)
    return y_pred

def _to_string(tree, indent="| "):
    """
    Menghasilkan representasi dalam bentuk string dari struktur Decision Tree.

    Args:
        tree (Tree): Objek pohon keputusan yang akan direpresentasikan.
        indent (str, optional): Indentasi yang digunakan untuk setiap level dalam representasi. Default: "| ".

    Returns:
        str: Representasi dalam bentuk string dari struktur Decision Tree.

    """
    if tree.leaf:
        text_to_print = f'Pred: {tree.val:.2f}'

    else:
        dec = f"feature_{tree.fitur} <= {tree.thresh:.2f}?"

        true_branch = indent + 'T: ' + _to_string(tree = tree.child_left,
                                                  indent = indent + '| ')

        false_branch = indent + 'T: ' + _to_string(tree = tree.child_right,
                                                  indent = indent + '| ')
        
        text_to_print = dec + '\n' + true_branch + '\n' + false_branch
    
    return text_to_print

class DecisionTreeBase:
    """
    Fungsi yang digunakan untuk melakukan Base Decision Tree untuk Regressor & Classifier,

        - max_depth                 = Maksimal kedalaman tree yang dikembangkan
        - criteria                  = {gini, entropy, log-loss}
        - impurity_reduction_min    = Reduksi impurity yang minimal untuk mengembangkan tree
        - sample_split_min          = Jumlah sample split minimal untuk split node
        - sample_leaf_min           = Jumlah sample leaf minimal untuk split node
        - alpa                      = Cost function tree pruning

    """
    def __init__(
        self,
        max_depth,
        criteria,
        impurity_reduction_min,
        sample_split_min,
        sample_leaf_min,
        alpa = 0.0
    ):
        self.max_depth = max_depth
        self.criteria = criteria
        self.impurity_reduction_min = impurity_reduction_min
        self.sample_split_min = sample_split_min
        self.sample_leaf_min = sample_leaf_min
        self.alpa = alpa

    def _most_split(self, X, y):
        """
        Fungsi yang digunakan untuk mencari fitur dan threshold dengan split paling optimal.

        Args:
            X: Data fitur.
            y: Data target.

        Returns:
            most_feature: Fitur dengan split paling optimal.
            most_thresh: Threshold dengan split paling optimal.
        """
        # Butuh minimal sample_split_min untuk split node
        n_shape = len(y)
        if n_shape < self.sample_split_min:
            return None, None
        
        # Inialisasi Decision Tree
        parent_data = np.column_stack((X, y))
        most_gain = 0.0
        most_feature, most_thresh = None, None
        for fitur_i in range(self.n_fitur):
            # Mengambil data dari fitur yang terpilih
            X_i = X[:, fitur_i]

            # Mencari threshold yang mungkin untuk splitting
            thresh = _split(data = X_i)

            # Melakukan iterasi untuk mencari spliting terbaik
            for i in range(len(thresh)):
                # Melakukan split pada root parent
                child_left, child_right = _split_data(data = parent_data,
                                                      fitur = fitur_i,
                                                      thresh = thresh[i])
                  
                # Mengambil output dari children
                y_left = child_left[:, self.n_fitur:]
                y_right = child_right[:, self.n_fitur:]
            
                # Menghitung penurunan impurity
                kond_1 = len(y_left) >= self.sample_leaf_min
                kond_2 = len(y_right) >= self.sample_leaf_min
            
                # Melakukan update terhadap most gain enggunakan calculate reduction impurity
                if kond_1 and kond_2:
                    present_gain = self._calculate_reduction_impurity(y,
                                                                    y_left,
                                                                    y_right)
                    if present_gain > most_gain:
                        most_gain = present_gain
                        most_feature = fitur_i 
                        most_thresh = thresh[i]

            # Return most_feature dan most threshold
        if most_gain >= self.impurity_reduction_min:
            return most_feature, most_thresh
        
        else:
            return None, None
        
    def _grow_tree(self, X, y, depth=0):
        """
        Fungsi yang digunakan untuk mengembangkan pohon keputusan secara rekursif.

        Args:
            X: Data fitur.
            y: Data target.
            depth: Kedalaman saat ini dalam pengembangan pohon.

        Returns:
            node: Node pohon keputusan.
        """
        # Membuat node untuk leaf atau node internal
        impurity_node = self._eval_impurity(y)
        val_node = self._calc_leaf_val(y)
        node = Tree(
             val = val_node,
             impurity = impurity_node,
             leaf = True,
             n_samples = len(y)
        )

        # Lakukan split secara rekursif hingga mencapai max_depth
        if self.max_depth is None:
             kondisi = True
        else:
             kondisi = depth <  self.max_depth

        if kondisi:
            # Mencari splitting terbaik
            fitur_i, thresh_i = self._most_split(X, y)

            if fitur_i is not None:
                # Split data
                data = np.column_stack((X, y))
                left_data, right_data = _split_data(data = data,
                                                    fitur = fitur_i,
                                                    thresh = thresh_i)
                # Ambil data X dan y
                X_data_left = left_data[:, :self.n_fitur]
                y_data_left = left_data[:, self.n_fitur:]
                X_data_right = right_data[:, :self.n_fitur]
                y_data_right = right_data[:, self.n_fitur:]

                # Menumbuhkan pohon
                node.fitur = fitur_i
                node.thresh = thresh_i
                node.child_left = self._grow_tree(X_data_left, y_data_left, depth+1)
                node.child_right = self._grow_tree(X_data_right, y_data_right, depth+1)
                node.leaf = False

        return node
                
    def _calculate_reduction_impurity(self, parent_data, child_left, child_right):
        """
        Fungsi yang digunakan untuk menghitung penurunan impuritas dengan split tertentu.

        Args:
            parent_data: Data pada node parent.
            child_left: Data pada child node kiri.
            child_right: Data pada child node kanan.

        Returns:
            impurity_reduction: Penurunan impuritas dengan split tertentu.
        """
        # Menghitung jumlah sample
        N = self.n_samples
        N_T = len(parent_data)
        N_t_L = len(child_left)
        N_t_R = len(child_right)

        # Menghitung nilai impurity
        I_parent = self._eval_impurity(parent_data)
        I_children_left = self._eval_impurity(child_left)
        I_children_right = self._eval_impurity(child_right)

        # Menghitung bobot dari impurity
        impurity_reduction = I_parent \
                            - (N_t_R / N_T) * I_children_right \
                            - (N_t_L / N_T) * I_children_left
        
        impurity_reduction *= (N_T/N)

        return impurity_reduction

    def _tree_pruning(self, tree=None):
        """
        Fungsi yang digunakan untuk melakukan pruning pada pohon keputusan.

        Args:
            tree: Pohon keputusan yang akan dipruning (default: None).
        """
        if not tree:
            tree = self.tree_

        if tree.leaf:
            pass
        else:
            self._tree_pruning(tree.child_left)
            self._tree_pruning(tree.child_right)

            if tree.child_right.leaf ==  False and tree.child_left.leaf == False:
                n_true = tree.child_left.n_samples
                n_false = tree.child_right.n_samples

                p = n_true / (n_true + n_false)
                delta = tree.impurity - p*tree.child_left.impurity - (1-p)*tree.child_right.impurity
                if delta < self.alpa:
                    tree.child_left, tree.child_right = None, None
                    tree.thresh = None
                    tree.fitur = None
                    tree.leaf = True

    def _export_tree(self):
        print("Decision Tree")
        print("-------------")
        print(_to_string(tree=self.tree_))
                     
    def _predict_val(self, X, tree=None):
        """
        Fungsi yang digunakan untuk melakukan prediksi nilai target berdasarkan pohon keputusan.

        Args:
            X: Data fitur.
            tree: Pohon keputusan yang digunakan untuk prediksi (default: None).

        Returns:
            predicted_val: Nilai target yang diprediksi.
        """
        # Melakukan pendefinisian terhadap tree
        if tree is None:
            tree = self.tree_
        
        # Lakukan pengecekan apakah terdapat leaf atau tidak
        if tree.leaf:
            # Mengembalikan predicted value yang ada pada leaf
            return tree.val
        else:
            # Apabila terdapat pada branch, lakukan pemilihan fitur
            fitur_val = X[:, tree.fitur]

            # Selanjutnya, tentukan branch yang diikuti
            if fitur_val <= tree.thresh:
                branch = tree.child_left
            else:
                branch = tree.child_right

            return self._predict_val(X, branch)
        
    def fit(self, X, y):
        """
        Fungsi yang digunakan untuk melatih model Decision Tree.

        Args:
            X: Data fitur.
            y: Data target.
        """
        # Lakukan konversi array terhadap input
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Melakukan ektraksi terhadap ukuran data
        self.n_samples, self.n_fitur = X.shape

        # Melakukan pengembangan Tree
        self.tree_ = self._grow_tree(X, y)

        # Melakukan tree pruning
        self._tree_pruning()

    def predict(self, X):
        """
        Fungsi yang digunakan untuk melakukan prediksi nilai target menggunakan model Decision Tree.

        Args:
            X: Data fitur.

        Returns:
            y: Nilai target yang diprediksi.
        """
        
        # Melakukan konversi terhadap input data
        X = np.array(X).copy()

        # Melakukan prediksi
        y = [self._predict_val(sample.reshape(1, -1)) for sample in X]

        return y
    
class DecisionTreeClassifier(DecisionTreeBase):
    """
    Decision Tree Classifier untuk memodelkan masalah klasifikasi.

    Args:
        criteria (str, optional): Kriteria yang digunakan untuk pemilihan atribut pemisah. Default: "gini".
        max_depth (int or None, optional): Kedalaman maksimum dari pohon keputusan. None berarti tidak ada batasan kedalaman. Default: None.
        sample_split_min (int, optional): Jumlah minimum sampel yang diperlukan untuk melakukan split pada node. Default: 2.
        sample_leaf_min (int, optional): Jumlah minimum sampel yang diperlukan pada leaf node. Default: 1.
        impurity_reduction_min (float, optional): Jumlah minimum pengurangan impurity yang diperlukan untuk melakukan split. Default: 0.0.
        alpha (float, optional): Parameter alpha untuk regularisasi dalam perhitungan impurity. Default: 0.0.

    """
    def __init__(
        self,
        criteria = "gini",
        max_depth = None,
        sample_split_min = 2,
        sample_leaf_min = 1,
        impurity_reduction_min = 0.0,
        alpa = 0.0
    ):
        super().__init__(
            criteria = criteria,
            max_depth = max_depth,
            sample_leaf_min = sample_leaf_min,
            sample_split_min = sample_split_min,
            impurity_reduction_min = impurity_reduction_min,
            alpa = alpa
        )
        
    def fit(self, X, y):
        """
        Melakukan pelatihan model Decision Tree Classifier.

        Args:
            X (array-like): Data fitur pelatihan.
            y (array-like): Data target pelatihan.

        """
        # Melakukan inisialisasi solver
        self._eval_impurity = CRITERIA_CLF[self.criteria]
        self._calc_leaf_val = _calculate_majority_vote

        super(DecisionTreeClassifier, self).fit(X, y)

class DecisionTreeRegressor(DecisionTreeBase):
    """
    Decision Tree Regressor untuk memodelkan masalah regresi.

    Args:
        criteria (str, optional): Kriteria yang digunakan untuk pemilihan atribut pemisah. Default: "squared_error".
        max_depth (int or None, optional): Kedalaman maksimum dari pohon keputusan. None berarti tidak ada batasan kedalaman. Default: None.
        sample_split_min (int, optional): Jumlah minimum sampel yang diperlukan untuk melakukan split pada node. Default: 2.
        sample_leaf_min (int, optional): Jumlah minimum sampel yang diperlukan pada leaf node. Default: 1.
        impurity_reduction_min (float, optional): Jumlah minimum pengurangan impurity yang diperlukan untuk melakukan split. Default: 0.0.
        alpha (float, optional): Parameter alpha untuk regularisasi dalam perhitungan impurity. Default: 0.0.

    """
    def __init__(
        self,
        criteria = "squared_error",
        max_depth = None,
        sample_split_min = 2,
        sample_leaf_min = 1,
        impurity_reduction_min = 0.0,
        alpa = 0.0
    ):
        super().__init__(
            criteria = criteria,
            max_depth = max_depth,
            sample_leaf_min = sample_leaf_min,
            sample_split_min = sample_split_min,
            impurity_reduction_min = impurity_reduction_min,
            alpa = alpa
        )

    def fit(self, X, y):
        """
        Melakukan pelatihan model Decision Tree Regressor.

        Args:
            X (array-like): Data fitur pelatihan.
            y (array-like): Data target pelatihan.

        """
        # Melakukan inisialisasi solver
        self._eval_impurity = CRITERIA_REG[self.criteria]
        self._calc_leaf_val = _calculate_average_vote

        super(DecisionTreeRegressor, self).fit(X, y)