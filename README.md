# Decision Tree Model From Scratche
## Latar Belakang
Dalam proyek ini, saya mengembangkan algoritma Decision Tree tanpa menggunakan library atau framework yang sudah ada, tetapi membangunnya dari awal menggunakan bahasa pemrograman Pythons.

Decision Tree adalah salah satu algoritma machine learning yang populer dan digunakan untuk tugas klasifikasi dan regresi. Tujuannya adalah untuk membuat model prediksi yang dapat mempelajari pola dari data pelatihan dan menghasilkan keputusan atau prediksi berdasarkan fitur-fitur yang ada. Dalam pengantar ini, saya akan menjelaskan langkah-langkah utama dalam membangun Decision Tree dari awal. Pertama, kita perlu mempersiapkan data pelatihan yang terdiri dari contoh-contoh dengan fitur-fitur dan label kelas yang terkait.

Selanjutnya, kita akan memilih atribut pemisah yang paling informatif untuk membangun node-node dalam Decision Tree. Proses pemilihan ini dapat dilakukan menggunakan kriteria seperti information gain, indeks Gini, atau pengurangan kesalahan klasifikasi. Setelah pemilihan atribut pemisah, kita akan membagi data pelatihan menjadi subset yang lebih kecil berdasarkan nilai-nilai atribut pada setiap node. Langkah ini akan terus diulangi untuk setiap subset data yang dihasilkan, sehingga terbentuklah struktur decision tree.

Selama proses pembangunan Decision Tree, kita juga perlu memperhatikan kondisi berhenti yang menghentikan proses splitting dan pembentukan node baru. Kondisi berhenti ini dapat ditentukan berdasarkan kedalaman maksimum (max_depth), jumlah minimum sampel dalam setiap node, atau ketika tidak ada atribut pemisah yang tersisa. Setelah Decision Tree selesai dibangun, kita dapat menggunakan model ini untuk melakukan prediksi pada data baru. Model Decision Tree akan mengikuti cabang-cabang berdasarkan fitur-fitur pada data uji dan menghasilkan prediksi berdasarkan label kelas yang terkait dengan leaf yang dicapai.

Melalui pengantar ini, saya berharap dapat memberikan pemahaman dasar tentang langkah-langkah yang terlibat dalam membangun Decision Tree dari awal. Algoritma Decision Tree sangat berguna dalam memecahkan masalah klasifikasi dan regresi, dan pemahaman tentang bagaimana mereka bekerja dari dasar dapat menjadi landasan yang kuat dalam memahami algoritma machine learning lainnya.

## Keuntungan dan Kelemahan Model
Dibandingkan dengan metode yang lebih tradisional, decision tree untuk regresi dan klasifikasi memberikan beberapa manfaat berikut:

* Beberapa orang berpikir bahwa decision tree, dibandingkan dengan metode regresi dan klasifikasi lainnya, lebih mirip dengan pengambilan keputusan manusia.
Orang-orang dapat dengan mudah memahami decision tree. Sebenarnya, decision tree membutuhkan penjelasan yang lebih sedikit dibandingkan dengan regresi linear.
* Decision tree dengan mudah mengatasi prediktor kualitatif dan tidak memerlukan penggunaan variabel dummy.
* Decision tree dapat direpresentasikan secara grafis dan cukup sederhana untuk dipahami oleh non-pakar (terutama jika pohon tersebut kecil).
Namun, terdapat beberapa kekurangan dalam menggunakan model decision tree, antara lain:

* Sayangnya, dibandingkan dengan beberapa metode regresi dan klasifikasi lainnya, decision tree biasanya tidak memiliki tingkat akurasi prediksi yang sama.
Decision tree mungkin juga tidak sangat tangguh. 
* Dengan kata lain, perubahan kecil pada data dapat memiliki dampak besar pada bentuk akhir decision tree yang diprediksi. Namun, kinerja prediksi decision tree dapat signifikan meningkat dengan menggabungkan banyak decision tree menggunakan teknik seperti bagging, random forests, dan boosting.
## Komponen Pembelajaran
Bagian dari Algoritma
* Root Node: Node pertama dalam decision tree. Ini mewakili fitur atau atribut yang paling penting dalam pembagian data.
* Node Internal: Node dalam decision tree yang tidak termasuk root node atau leaf. Node internal mewakili keputusan berdasarkan fitur-fitur atau atribut-atribut yang lebih spesifik.
* Leaf Node: Node-terminal dalam decision tree yang tidak memiliki anak. Node leaf mewakili hasil atau label kelas yang dihasilkan oleh decision tree.
* Cabang (Branch): Cabang-cabang dalam decision tree yang menghubungkan node-node. Cabang mewakili penghubung antara fitur atau atribut dengan keputusan atau sub-tree berikutnya.
* Test Attribut: Fitur atau atribut yang digunakan untuk membagi data menjadi kelompok-kelompok yang lebih kecil dalam setiap node.
* Pembagian Data (Data Partitioning): Proses memisahkan data ke dalam subset yang lebih kecil berdasarkan nilai-nilai fitur atau atribut yang digunakan dalam pengujian.
* Kriteria Pemilihan Test Attribut: Kriteria yang digunakan untuk memilih atribut terbaik untuk membagi data, seperti information gain, indeks Gini, atau pengurangan kesalahan klasifikasi.
* Pruning: Proses menghapus cabang-cabang yang tidak relevan atau node leaf yang tidak memberikan keuntungan informasi yang signifikan.
### Hipotesis
Set seluruh splitting variable
* Pilih fitur terbaik untuk prediktor kategori
* Hitung classification error pada setiap keputusan
* Hitung prediksi dan kesalahan pada setiap node
* Pilih fitur terbaik untuk prediktor numerik
* Memilih threshold yang membagi data menjadi dua region
* Memprediksi output dari setiap region (klasifikasi atau regresi)
* Hitung kesalahan klasifikasi untuk setiap region
* Hitung perbaikan (improvement) pada setiap fitur
* Pilih fitur terbaik berdasarkan perbaikan tertinggi
### Komponen Pembelajaran
* Subset dari data X pada node tree
* Pada setiap fitur di X, split data berdasarkan fitur p dan hitung kesalahan klasifikasi
* Pilih fitur-p* dengan perbaikan (improvement) tertinggi
* Hitung impurity untuk mengukur ketidakseragaman data
* Hitung information gain untuk membandingkan impurity sebelum dan setelah split
* Gunakan metode pruning untuk menghapus cabang atau node leaf yang tidak relevan
### Hyperparameter
* max_depth: Membatasi jumlah splitting dan mengontrol kompleksitas pohon.
* sample_leaf_min: Menentukan jumlah minimum sampel yang diperlukan dalam sebuah node leaf pada decision tree. Jika jumlah sampel dalam node leaf turun di bawah ambang batas ini, pemisahan lebih lanjut dihentikan, dan node tersebut menjadi node leaf.
* sample_split_min: Menentukan jumlah minimum sampel yang diperlukan untuk melakukan pemisahan pada suatu node. Jika jumlah sampel pada suatu node kurang dari ambang batas ini, proses pemisahan dihentikan, dan node tersebut menjadi node leaf.
* impurity_reduction_min: Mengatur threshold untuk jumlah minimum pengurangan impurity yang diperlukan untuk melakukan splitting. Jika pengurangan impurity yang dicapai * * oleh pemisahan potensial berada di bawah threshold ini, splitting tidak dianggap, dan node tersebut menjadi node leaf.
### Input
* X: Dataset training input
* y: Dataset training output
Kriteria
  Untuk kasus regresi:
* Menggunakan kriteria minimisasi jumlah kuadrat untuk memilih pemisah dan titik pemisahan terbaik.
* Menggunakan metode greedy untuk menentukan pemisah dan titik pemisahan yang memaksimalkan penurunan impurity.
  Untuk kasus klasifikasi:
* Menggunakan kriteria seperti miss classification error, Gini index, atau cross-entropy untuk mengukur ketidakmurnian node dan memilih pemisah terbaik.
### Output
Set fitur dan threshold yang digunakan untuk pemisahan (splitting)
### Kriteria penghentian (stopping criteria) sebagai hyperparameter
* max_depth: Membatasi jumlah splitting dan mengontrol kompleksitas pohon.
* sample_leaf_min: Menentukan jumlah minimum sampel yang diperlukan dalam sebuah node leaf.
* sample_split_min: Menentukan jumlah minimum sampel yang diperlukan untuk melakukan pemisahan pada suatu node.
* impurity_reduction_min: Menentukan jumlah minimum pengurangan impurity yang diperlukan untuk melakukan splitting.
## Kompleksitas dan Kesesuaian Subtrees
Penggunaan parameter penyetelan α untuk mengontrol kompleksitas pohon.
Subtrees T ⊆ T0 yang menghasilkan kesesuaian (fit) terbaik dengan data pelatihan dipilih menggunakan cost complexity pruning.
Subtrees T dipilih berdasarkan α yang meminimalkan C(T).
Keseimbangan antara ukuran pohon dan kemampuan pohon untuk cocok dengan data dikendalikan oleh parameter α.
Subtree T0 lengkap diperoleh saat α = 0, sementara subtree T yang lebih kecil dihasilkan dengan α yang lebih besar.
Penggunaan set validasi atau cross validation untuk memilih nilai α yang tepat.
Subtree yang sesuai dengan α kemudian diperoleh dari set data lengkap.
### Referensi
* [1] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification and Regression Trees", Wadsworth, Belmont, CA, 1984
* [2] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical Learning", Springer, 2009.
* [3] Gareth James, et. al. An Introduction to Statistical Learning
* [4] Lecture Notes :
https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/f5678de0e329ce097fc6ec6182ebaea2_MIT15_097S12_lec08.pdf
https://cs229.stanford.edu/notes2021fall/lecture11-decision-trees.pdf
