# **Water-Potability-Assessment**
![safe-water-filtration](https://github.com/Sulbae/Water-Potability-Assessment/blob/main/assets/banner/safe-water-filtration%202.jpg)

---
## **List of Contents**
  - [Background](#Background)
  - [Objective](#Objective)
  - [Scpoe](#Scope)
  - [Dataset Overview](#Dataset-Overview)
  - [Data Preparation](#Data-Preparation)
    - [Data Assessment](#Data-Assessment)
    - [Data Cleaning](#Data-Cleaning)
    - [Column Description](#Column-Description)
    - [Explore](#Explore)
  - [Data Preparation](#Data-Preparation#)
    - [Features Engineering](#Feature-Engineering)
    - [Data Normalization](#Data-Normalization)
    - [Data Exploration](#Data-Exploration)
    - [Data Preprocessing](#Data-Preprocessing)
  - [Model Development](#Model-Development)
    - [Classifier Model](#Classifier-Model)
    - [Anomali Detection Model](#Anomaly-Detection-Model)
  - [Risk Assessment](#Risk-Assessment)
  - [Inference](#Inference)
  - [Conclusion](#Conclusion)
  - [Project Evaluation & Future Improvement](#Project-Evaluation-&-Future-Improvement)
  - [References](#References)

---
## **Background**
   Berdasarkan survei Badan Pusat Statistik (BPS) tahun 2024, 9 dari 10 rumah tangga di Indonesia memiliki akses air minum yang layak. Namun, hanya 11,9% yang memiliki akses air minum yang benar-benar aman sesuai standar kesehatan. Survei kualitas air rumah tangga menunjukkan bahwa persentase sampel air minum yang terkontaminasi bakteri seperti E. coli bisa mencapai sekitar 70%, yang menunjukkan potensi risiko kesehatan tinggi [1]. Kontaminasi mikroorganisme patogen seperti ini maupun senyawa berbahaya lain yang terkandung dalam air merupakan ancaman nyata terhadap keamanan para konsumen. Jika kondisi ini terus berlanjut dan tak kunjung membaik, maka kebutuhan air bersih layak minum akan menjadi tantangan nyata bagi masyarakat. Oleh karena itu, pemerintah melalui Peraturan Presiden Republik Indonesia Nomor 12 Tahun 2025 tentang Rencana Pembangunan Jangka Menengah Nasional (RPJMN) 2025-2029 telah membuat beberapa target untuk memastikan ketersediaan air bersih untuk penduduk Indonesia. Salah satu targetnya yaitu meningkatkan akses rumah tangga perkotaan terhadap air siap minum perpipaan hingga 51,36% [2].
   
   Sejalan dengan itu, ADEM Indonesia (perusahaan fiktif), sebagai sebuah perusahaan layanan penyediaan air minum (SPAM) akan memiliki peran serta dalam merealisasikan program tersebut. Dengan demikian, perushaan memiliki tanggung jawab untuk memastikan bahwa air yang didistribusikan kepada masyarakat telah memenuhi standar kualitas, keamanan, dan kesehatan secara konsisten, akurat, dan berkelanjutan. Penggunaan sumber air baku yang berasal dari sungai, waduk, danau, dan air tanah, tentu memiliki tingkat kontaminasi yang berbeda-beda, baik secara fisik, kimia, maupun biologis. Hal ini menjadi tantangan yang paling sering dihadapi dalam operasional pengolahan air. Ditambah lagi perubahan musim, limpasan limbah domestik, aktivitas industri, dan degradasi kualitas lingkungan memperbesar risiko fluktuasi kualitas air.
   
   Sebagai respon terhadap tantangan tersebut, perusahaan memutuskan untuk mengembangkan sistem monitoring kelayakan air terintegrasi. Salah satu komponen yang perlu dikembangkan adalah Water Potability Classifier berbasis Machine Learning. Komponen ini dirancang untuk membantu meningkatkan kinerja perusahaan dalam mengolah multi-parameter kualitas air (fisik, kimia, dan biologis) secara simultan dan menghasilkan klasifikasi potabilitas air secara cepat dan sistematis.

## Objective
* Mengembangkan model machine learning untuk monitoring potabilitas/kelayakan air berdasarkan data hasil uji lab kualitas air yang dimiliki perusahaan.
![system_workflow](https://github.com/Sulbae/Water-Potability-Assessment/blob/12627c4d22f192855a8c9aa3d11d88a0525961e2/assets/workflow/system%20workflow.png)

## Scope
Pengembangan tahap awal, yaitu: 
   1) membuat model **klasifikasi potabilitas air**,
   2) membuat model **deteksi anomali kualitas air**.

## **Dataset Overview**
Dataset yang digunakan adalah water_potability.csv (dataset publik yang tersedia di platform Kaggle) yang berisi data metrik kualitas air dari 3276 badan air yang berbeda. Fitur-fitur yang tersedia dalam dataset ditampilkan pada Tabel 1.
| No | Nama | Tipe Data | Deskripsi |
|----|------|-----------|-----------|
| 1 | pH | float64 | PH merupakan parameter penting dalam mengevaluasi keseimbangan asam-basa air. PH juga merupakan indikator status air yang bersifat asam atau basa. WHO telah merekomendasikan batas pH maksimum yang diizinkan dari 6,5 hingga 8,5. Kisaran pH yang diteliti saat ini adalah 6,52–6,83 yang berada dalam kisaran standar WHO. |
| 2 | Hardness | float64 | Kesadahan terutama disebabkan oleh garam kalsium dan magnesium. Garam-garam ini terlarut dari endapan geologis yang dilalui air. Lamanya waktu air bersentuhan dengan bahan yang menghasilkan kesadahan membantu menentukan seberapa banyak kesadahan yang ada dalam air mentah. Kesadahan awalnya didefinisikan sebagai kapasitas air untuk mengendapkan sabun yang disebabkan oleh Kalsium dan Magnesium. |
| 3 | Solids (Total Dissolved Solids - TDS) | float64 | Air memiliki kemampuan untuk melarutkan berbagai macam mineral atau garam anorganik dan beberapa mineral atau garam organik seperti kalium, kalsium, natrium, bikarbonat, klorida, magnesium, sulfat, dll. Mineral-mineral ini menghasilkan rasa yang tidak diinginkan dan warna yang encer pada air. Ini adalah parameter penting untuk penggunaan air. Air dengan nilai TDS yang tinggi menunjukkan bahwa air tersebut sangat bermineral. Batas yang diinginkan untuk TDS adalah 500 mg/l dan batas maksimum adalah 1000 mg/l yang ditentukan untuk tujuan minum. |
| 4 | Chloramines | float64 | Klorin dan kloramina merupakan disinfektan utama yang digunakan dalam sistem air publik. Kloramina paling sering terbentuk ketika amonia ditambahkan ke klorin untuk mengolah air minum. Kadar klorin hingga 4 miligram per liter (mg/L atau 4 bagian per juta (ppm)) dianggap aman dalam air minum. |
| 5 | Sulfate | float64 | Sulfat adalah zat alami yang ditemukan dalam mineral, tanah, dan bebatuan. Zat ini terdapat di udara sekitar, air tanah, tanaman, dan makanan. Penggunaan utama sulfat secara komersial adalah dalam industri kimia. Konsentrasi sulfat dalam air laut sekitar 2.700 miligram per liter (mg/L). Konsentrasinya berkisar antara 3 hingga 30 mg/L dalam sebagian besar persediaan air tawar, meskipun konsentrasi yang jauh lebih tinggi (1000 mg/L) ditemukan di beberapa lokasi geografis. |
| 6 | Conductivity | float64 | Air murni bukanlah konduktor arus listrik yang baik, melainkan isolator yang baik. Peningkatan konsentrasi ion meningkatkan konduktivitas listrik air. Secara umum, jumlah padatan terlarut dalam air menentukan konduktivitas listrik. Konduktivitas listrik (EC) sebenarnya mengukur proses ionik suatu larutan yang memungkinkannya mengalirkan arus. Menurut standar WHO, nilai EC tidak boleh melebihi 400 μS/cm. |
| 7 | Organic Carbon | float64 | Total Organic Carbon (TOC) di sumber air berasal dari bahan organik alami (NOM) yang membusuk serta sumber sintetis. TOC adalah ukuran jumlah total karbon dalam senyawa organik dalam air murni. Menurut US EPA < 2 mg/L sebagai TOC dalam air olahan/air minum, dan < 4 mg/Lit dalam sumber air yang digunakan untuk pengolahan. |
| 8 | Trihalomethanes | float64 | THM adalah zat kimia yang dapat ditemukan dalam air yang diolah dengan klorin. Konsentrasi THM dalam air minum bervariasi menurut tingkat bahan organik dalam air, jumlah klorin yang dibutuhkan untuk mengolah air, dan suhu air yang diolah. Kadar THM hingga 80 ppm dianggap aman dalam air minum. |
| 9 | Turbidity | float64 | Kekeruhan air bergantung pada kuantitas zat padat yang ada dalam keadaan tersuspensi. Kekeruhan merupakan ukuran sifat air yang memancarkan cahaya dan pengujian ini digunakan untuk menunjukkan kualitas pembuangan limbah terhadap zat koloid. Nilai kekeruhan rata-rata yang diperoleh untuk Kampus Wondo Genet (0,98 NTU) lebih rendah dari nilai yang direkomendasikan WHO yaitu 5,00 NTU. |\n",
| 10 | Potability | int64 | Menunjukkan apakah air aman untuk dikonsumsi manusia, di mana 1 berarti Layak Minum dan 0 berarti Tidak Layak Minum. |

Sumber: [Lihat](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

## **Data Preparation**
### **Data Assessment**
Dari total 3276 baris data, ditemukan sebanyak **1434** nilai null/missing value pada dataset. Sementara itu, hasil pemeriksaan menunjukkan tidak ada duplikasi data pada dataset.

### **Data Cleaning**
#### **Missing Value Handling**
Nilai null/minssing value pada dataset ditemukan pada 3 variabel (dalam hal ini data parameter kualitas air) yaitu pH (491 null), Sulfate (781 null), dan Trihalomethanes (162 null). Untuk mengatasi kekosongan data ini, maka dilakukan imputasi data menggunakan nilai median untuk menjaga struktur distribusi data. 

### **Data Exploration**
#### 1) Cek Distribusi Data
   ![histogram-distribusi-data](https://github.com/Sulbae/Water-Potability-Assessment/blob/708e8b6e1cb1403bfcd9e2c81d2ae869b7c9da90/assets/EDA/Distribusi%20Data%20Fitur.png)

   * Data setiap parameter terdistribusi normal, kecuali parameter Solids yang sedikit condong ke kiri / positive skewed (Mean > Median dan nilai max = 3x Median)
   * Berdasarkan kelas targetnya, ternyata proporsi dataset diketahui tidak seimbang (imbalance). Untuk mengatasi ini, maka perlu penerapan weight balance pada proses pelatihan model klasifikasi.

#### 2) Cek Outlier
   ![boxplot-outlier](https://github.com/Sulbae/Water-Potability-Assessment/blob/c6bf27c09814fc2e0b5e308ef1b01dd6a54a933b/assets/EDA/Outlier.png)

   * Terdapat outlier pada semua parameter. Hal ini dapat dinilai wajar terjadi apabila mempertimbangkan kejadian fluktuasi kualitas air akibat berbagai faktor lingkungan. Informasi outlier akan berguna untuk pengembangan model deteksi anomali.
   
#### 3) Cek Korelasi
   ![heatmap-korelasi](https://github.com/Sulbae/Water-Potability-Assessment/blob/c6bf27c09814fc2e0b5e308ef1b01dd6a54a933b/assets/EDA/heatmap%20korelasi.png)

   * Semua nilai korelasi terhadap Potability sangat kecil (mendekati 0). Artinya, korelasi linear sangat lemah. Potabilitas ditentukan melalui kombinasi multi-variabel. Dengan demikian, model linear tidak cocok untuk data ini.
   
### **Data Preprocessing**
#### Preprocessing Pipeline
![preprocessing-pipeline](https://github.com/Sulbae/Water-Potability-Assessment/blob/b71b4c59d13bb7115507a260949fb846bdf4147f/assets/workflow/preprocessing_pipeline.png)

Berdasarkan hasil temuan dari eksplorasi data, maka dilakukan Preprocessing Data dengan tahapan:
1) Imputasi

   Imputasi dilakukan dengan memberikan nilai baru sebagai pengganti nilai null / data yang hilang. Dengan mempertimbangkan jumlah `null` yang cukup banyak, maka nilai yang digunakan untuk imputasi adalah nilai median. Median dipilih karena sifatnya yang netral sehingga tidak berpengaruh signifikan terhadap distribusi data.
   
3) Scaling

   Standarisasi nilai skala dilakukan untuk menyeragamkan rentang data seluruh parameter (pada data pelatihan) sehingga data memiliki rentang yang tidak terlalu berbeda signifikan. Hal ini dilakukan untuk mencegah bias dari pelatihan model deteksi anomali karena model deteksi anomali sensitif terhadap nilai/skala fitur.

   `StandardScaler` dipilih sebagai scaler karena distribusi seluruh data parameter kualitas air sudah normal/mendekati normal. scaler ini bekerja dengan melakukan transformasi nilai data agar memiliki rata-rata **(_mean_) = 0** dan **simpangan baku (_standar deviasi_) = 1**. Secara matematis, setiap nilai fitur ditransformasikan seperti berikut:

$$
x_{scaled} = \frac{x - \mu}{\sigma}
$$
  - Keterangan:
    - μ = mean (nilai rata-rata) fitur 
    - σ = standar deviasi fitur 

## **Model Development**
### **Classifier Model**
#### **Algorithm**
`RandomForestClassifier()` merupakan algoritma yang bekerja dengan menggabungkan banyak pohon keputusan (_decision tree_) yang akan menentukan hasil klasifikasi berdasarkan suara terbanyak.

Kelebihan:
* Akurasi & Stabilitas tinggi
* Mampu menangkap hubungan non-linear
* Tidak sensitif terhadap perbedaan skala fitur
* Relatif tahan terhadap outlier

Kekurangan:
* Model cenderung sulit diinterpretasikan secara detail.
* Waktu training dan inferensi lebih lambat dibanding model sederhana.
* Kurang cocok untuk kebutuhan analisis real-time dengan latensi rendah.
* Jika terlalu kompleks, ukuran model besar dan boros memori.

#### Parameter
1) n_estimator
   - Jumlah pohon keputusan (decision tree) yang dibangun dalam Random Forest.
   - Semakin banyak, prediksi lebih akurat & stabil.
   - Jika terlalu banyak, waktu pelatihan & inferensi semakin lama.
2) max_depth
   - Kedalaman (jumlah anak cabang) maksimum  setiap pohon keputusan yang dapat mengontrol kompleksitas model.
   - Nilai besar, model akan sangat detail. Ada, risiko overfitting.
   - Nilai kecil, model terlalu sederhana. Ada risiko underfitting.
3) min_samples_leaf
   - Jumlah minimum data dalam satu daun (leaf) pohon.
   - Semakin besar, model semakin smooth & stabil.
   - Dapat mengurangi efek noise dan outlier.
4) min_samples_split
   - Jumlah minimum data yang diperlukan agar suatu node bisa dipecah (split).
   - Nilai besar, pohon lebih sederhana.
   - Nilai kecil, pohon lebih detail.

#### **Training**
Pelatihan dilakukan menggunakan `GridSearch()` sehingga dapat melakukan hyperparameter tuning sekaligus dan menghasilkan _best_model_ dengan _best_parameter_. Metrik skor yang digunakan yaitu precision, recall, dan F1-score.

Konfigurasi Parameter:
- `N_ESTIMATORS_RANGE = np.linspace(10, 100, 3, dtype=int)`
- `MAX_DEPTH_RANGE = np.linspace(1, 50, 3, dtype=int)`
- `MIN_SAMPLES_LEAF = np.linspace(1, 10, 3, dtype=int)`
- `MIN_SAMPLES_SPLIT = np.linspace(2, 10, 3, dtype=int)`

Best Parameter:
- n_estimators = **100**
- max_depth = **50**
- min_samples_leaf = **1**
- min_samples_split = **2**

#### **Evaluation**
##### **Metrik Evaluasi**
a) Precision: Tingkat ketepatan prediksi positif yang dibuat model.

Formula matematis:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Keterangan:
- TP = True Positive - positif dan diprediksi positif
- FP = False Positive - negatif tapi diprediksi positif

b) Recall: Tingkat keberhasilan/kemampuan model dalam mendeteksi seluruh data positif yang sebenarnya.

Formula matematis:

$$
\text{Precision} = \frac{TP}{TP + FN}
$$

Keterangan:
- TP = True Positive - positif dan diprediksi positif
- FN = False Negative - positif tapi diprediksi negatif

c) F1-Score: 
   - Metrik yang menggabungkan precision dan recall. 
   - Digunakan ketika ingin keduanya memiliki bobot yang seimbang.

Formula matematis:

$$
\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

atau

$$
\text{F1-score} = \frac{2TP}{2TP + FP + FN}
$$

##### **Hasil Evaluasi**
Dalam pengembangan model klasifikasi Potabilitas Air ini, ditetapkan **_threshold_ probabilitas** prediksi sebesar **0.69** sehingga model akan lebih ketat dalam mengkategorikan air yang tidak layak.

1) Classification Report

![classification-report](https://github.com/Sulbae/Water-Potability-Assessment/blob/a688f27c024d4cac6c3b565ab8b2169274eccd07/assets/model%20evaluation/classification%20report.png)

* Precision
  - Dari semua sampel yang diprediksi tidak layak, **62%** memang benar tidak layak.
  - Semua sampel yang diprediksi layak memang benar layak.

* Recall
  - Dari semua air yang sebenarnya tidak layak, **100%** berhasil terdeteksi.
  - Dari semua sampel yang sebenarnya layak, hanya **5%** yang berhasil dikenali.

2) Confusion Matrix

![confusion-matrix-klasifikasi](https://github.com/Sulbae/Water-Potability-Assessment/blob/b71b4c59d13bb7115507a260949fb846bdf4147f/assets/model%20evaluation/confusion%20matrix%20classifier%20th-0.69.png)
* Menetapkan threshold klasifikasi = 0.69, sehingga model dapat bekerja lebih ketat dan tidak meloloskan air tidak layak dengan mudah. Hal ini dilakukan untuk menjamin keamanan/kesehatan pengguna.

### **Anomaly Detection Model**
#### **Algorithm**
`IsolationForest()` merupakan algoritma yang dapat mendeteksi data tidak normal dengan cara mengisolasi data tersebut. Data anomali lebih mudah dipisahkan daripada data normal sehingga data yang cepat terisolasi dapat dianggap anomali.

Kelebihan
* Tidak perlu data anomali berlabel
* Skalabel untuk dataset besar
* Relatif tahan terhadap noise
* Tidak bergantung pada distribusi data

Kekurangan
* Perlu post-analysis (penjelasan mengapa anomali terjadi)
* Sensitif terhadap parameter contamination
* Tidak cocok sebagai classifier, hanya sebagai pelengkap

#### **Parameter**
1) n_estimator
   - Jumlah isolation tree yang dibangun dalam Isolation Forest.
   - Semakin banyak, maka skor anomali lebih stabil.
   - Jika terlalu sedikit, hasil akan kurang stabil (fluktuatif).
   - Jika terlalu banyak, waktu pelatihan & inferensi semakin lama.
2) max_samples
   - Jumlah atau proporsi sampel data yang digunakan untuk membangun setiap pohon (tree).
   - Nilai besar, pola global lebih terlihat.
   - Nilai kecil, fokus ke pola lokal.
3) contamination
   - Perkiraan proporsi data yang dianggap anomali.
   - Menentukan threshold pemisahan data normal vs anomali.
   - Disesuaikan dengan kebutuhan analisis.
4) max_features
   - Jumlah fitur yang dipilih secara acak untuk setiap tree.
   - Dapat meningkatkan variasi antar tree.
   - Mencegah satu fitur dominan.
   - Meningkatkan generalisasi.

#### **Training**
Pelatihan model dilakukan menggunakan data Train tanpa label (anom_train = clf_X_train), namun diketahui sebelumnya telah difilter **hanya data non-potable**. Hal ini bertujuan untuk mendeteksi anomali pada prediksi non-potable yang sangat ketat akibat threshold yang cukup tinggi pada model klasifikasi.

Konfigurasi Parameter:
- n_estimators = **100**
- max_samples = **'auto'**
- contamination = **0,05**

#### **Evaluation**
1) Anomali Rate & Score

![anomali-score-distribution](https://github.com/Sulbae/Water-Potability-Assessment/blob/2f328b45c14f4cb1de6014b27bbbec3dde8e619a/assets/model%20evaluation/distribusi%20skor%20anomali.png)
* Setelah diuji terhadap data Test, diketahui model memiliki **Anomaly Rate sebesar 6,25%**.
* Kemudian, berdasarkan histogram terlihat bahwa **distribusi melebar** ⟶ model dapat membedakan data normal vs anomali.

2) Confusion matrix

![confusion-matrix-deteksi-anomali](https://github.com/Sulbae/Water-Potability-Assessment/blob/b71b4c59d13bb7115507a260949fb846bdf4147f/assets/model%20evaluation/confusion%20matrix%20anomali%20detection.png)

* Dari **400 sampel air tidak layak**, sebanyak **22 sampel** terdeteksi memiliki distribusi data parameter yang **tidak wajar**.
* Dari **256 sampel air layak**, sebanyak **19 sampel** terdeteksi memiliki distribusi data parameter yang **tidak wajar**.

## **Risk Assessment**

![risk_assessment](https://github.com/Sulbae/Water-Potability-Assessment/blob/78c3343ebf28eb0ef9537c4b0b8ece03f0acbf4f/assets/inference/risk%20assessment.png)

## **Inference**

![inference-platform](https://github.com/Sulbae/Water-Potability-Assessment/blob/2f328b45c14f4cb1de6014b27bbbec3dde8e619a/assets/inference/inference_streamlit_1.png)

![inference-result](https://github.com/Sulbae/Water-Potability-Assessment/blob/2f328b45c14f4cb1de6014b27bbbec3dde8e619a/assets/inference/inference_streamlit_2.png)

**Streamlit UI** digunakan sebagai platform penerima input data dalam bentuk formulir yang dapat diisi secara manual.
App: [Coba App](https://water-potability-assessment.streamlit.app/)

## **Conclusion**
_**Water Potability Risk Assessment System**_ berhasil mengimplementasikan pendekatan machine learning dengan kombinasi antara supervised classification dan semi-supervised anomaly detection. Dan mampu memberikan hasil analisis risk level serta rekomendasi berdasarkan data masukkan (input) melalui formulir yang dibuat menggunakan streamlit UI.

* Catatan:
  - **Sistem Risk Assessment** yang telah dikembangkan tidak berperan sebagai pengganti uji analisis laboratorium, melainkan sebagai **alat bantu (_tools_)** untuk mempermudah analisis lanjutan.
  - **Input manual** pada streamlit UI munkgin berbeda dengan distribusi data pada pelatihan model, sehingga lebih mudah terdeteksi sebagai anomali.

## **Project Evaluation & Future Improvement**
- Kalibrasi Threshold
  * Threshold potabilitas saat ini masih **bersifat statis** sehingga perlu dipantau secara berkala.
  * Pengembangan selanjutnya dapat dipertimbangkan **implementasi threshold yang lebih adaptif** dan disesuaikan dengan regulasi resmi.

- Analisis Penyebab Anomali
  * Saat ini output analisis hanya menampilkan **risk level & rekomendasi**.
  * Analisis dapat dikembangkan lebih detail seperti **analisis parameter penyebab anomali**. 

## **References**
[1] Ekowati, A. P., & Lusno, M. F. D . Analisis Capaian dan Tantangan Akses Air Minum Aman di Indonesia Menuju SDGS 6.1.1 . Jurnal Penelitian Inovatif, 5(2), 1707–1714. (2025). https://doi.org/10.54082/jupin.1538

[2] Alfathi, B. R. 50% Penduduk Indonesia Diprediksi Kekurangan Air Bersih pada 2050. Artikel Lingkungan, GoodStats. (2025). https://data.goodstats.id/statistic/50-penduduk-indonesia-diprediksi-kekurangan-air-bersih-pada-2050-nNSet
