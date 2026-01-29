# Water-Potability-Assessment
![safe-water-filtration](https://github.com/Sulbae/Water-Potability-Assessment/blob/main/assets/banner/safe-water-filtration%202.jpg)
## Background
   Berdasarkan survei Badan Pusat Statistik (BPS) tahun 2024, 9 dari 10 rumah tangga di Indonesia memiliki akses air minum yang layak. Namun, hanya 11,9% yang memiliki akses air minum yang benar-benar aman sesuai standar kesehatan. Survei kualitas air rumah tangga menunjukkan bahwa persentase sampel air minum yang terkontaminasi bakteri seperti E. coli bisa mencapai sekitar 70%, yang menunjukkan potensi risiko kesehatan tinggi [1]. Kontaminasi mikroorganisme patogen seperti ini maupun senyawa berbahaya lain yang terkandung dalam air merupakan ancaman nyata terhadap keamanan para konsumen. Jika kondisi ini terus berlanjut dan tak kunjung membaik, maka kebutuhan air bersih layak minum akan menjadi tantangan nyata bagi masyarakat. Oleh karena itu, pemerintah melalui Peraturan Presiden Republik Indonesia Nomor 12 Tahun 2025 tentang Rencana Pembangunan Jangka Menengah Nasional (RPJMN) 2025-2029 telah membuat beberapa target untuk memastikan ketersediaan air bersih untuk penduduk Indonesia. Salah satu targetnya yaitu meningkatkan akses rumah tangga perkotaan terhadap air siap minum perpipaan hingga 51,36% [2].
   
   Sejalan dengan itu, ADEM Indonesia (perusahaan fiktif), sebagai sebuah perusahaan layanan penyediaan air minum (SPAM) akan memiliki peran serta dalam merealisasikan program tersebut. Dengan demikian, perushaan memiliki tanggung jawab untuk memastikan bahwa air yang didistribusikan kepada masyarakat telah memenuhi standar kualitas, keamanan, dan kesehatan secara konsisten, akurat, dan berkelanjutan. Penggunaan sumber air baku yang berasal dari sungai, waduk, danau, dan air tanah, tentu memiliki tingkat kontaminasi yang berbeda-beda, baik secara fisik, kimia, maupun biologis. Hal ini menjadi tantangan yang paling sering dihadapi dalam operasional pengolahan air. Ditambah lagi perubahan musim, limpasan limbah domestik, aktivitas industri, dan degradasi kualitas lingkungan memperbesar risiko fluktuasi kualitas air.
   
   Sebagai respon terhadap tantangan tersebut, perusahaan memutuskan untuk mengembangkan sistem monitoring kelayakan air terintegrasi. Salah satu komponen yang perlu dikembangkan adalah Water Potability Classifier berbasis Machine Learning. Komponen ini dirancang untuk membantu meningkatkan kinerja perusahaan dalam mengolah multi-parameter kualitas air (fisik, kimia, dan biologis) secara simultan dan menghasilkan klasifikasi potabilitas air secara cepat dan sistematis.

## Objective
* Mengembangkan model machine learning untuk monitoring potabilitas/kelayakan air berdasarkan data hasil uji lab kualitas air yang dimiliki perusahaan. Bagaimana cara mengembangkan baseline model machine learning untuk klasifikasi potabilitas air?

## Scope
Pengembangan tahap awal, yaitu: 
   1) membuat model **klasifikasi potabilitas air**,
   2) membuat model **deteksi anomali kualitas air**.

## Dataset Overview
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

## Data Preparation
### Data Assessment
Dari total 3276 baris data, ditemukan sebanyak **1434** nilai null/missing value pada dataset. Sementara itu, hasil pemeriksaan menunjukkan tidak ada duplikasi data pada dataset.

### Data Cleaning
#### Missing Value Handling
Nilai null/minssing value pada dataset ditemukan pada 3 variabel (dalam hal ini data parameter kualitas air) yaitu pH (491 null), Sulfate (781 null), dan Trihalomethanes (162 null). Untuk mengatasi kekosongan data ini, maka dilakukan imputasi data menggunakan nilai median untuk menjaga struktur distribusi data. 

### Data Exploration
#### 1) Cek Distribusi Data
   ![histogram-distribusi-data](https://github.com/Sulbae/Water-Potability-Assessment/blob/708e8b6e1cb1403bfcd9e2c81d2ae869b7c9da90/assets/EDA/Distribusi%20Data%20Fitur.png)

#### 2) Cek Outlier
   ![boxplot-outlier](https://github.com/Sulbae/Water-Potability-Assessment/blob/c6bf27c09814fc2e0b5e308ef1b01dd6a54a933b/assets/EDA/Outlier.png)
   
#### 3) Cek Korelasi
   ![heatmap-korelasi](https://github.com/Sulbae/Water-Potability-Assessment/blob/c6bf27c09814fc2e0b5e308ef1b01dd6a54a933b/assets/EDA/heatmap%20korelasi.png)
   
### Data Preprocessing
#### Preprocessing Pipeline
#### 1) Classifier Model
..data split
..standarisasi

#### 2) Anomali Detection Model
..data split
..standarisasi

## Pengembangan Model
### 1) Classifier Model
#### Algorithm
RandomForestClassifier()
..jelaskan konsep kerja
..kelebihan
..kekurangan

#### Training
..hyperparameter tuning

#### Evaluation
..classification report
..confusion matrix

### 2) Anomali Detection Model
#### Algorithm
IsolationForest()
..jelaskan konsep kerja
..kelebihan
..kekurangan

#### Training
..hyperparameter tuning

#### Evaluation
..classification report
..confusion matrix

## Realisasi
### Deployment

### Inference

## Referensi
[1] Ekowati, A. P., & Lusno, M. F. D . Analisis Capaian dan Tantangan Akses Air Minum Aman di Indonesia Menuju SDGS 6.1.1 . Jurnal Penelitian Inovatif, 5(2), 1707–1714. (2025). https://doi.org/10.54082/jupin.1538

[2] Alfathi, B. R. 50% Penduduk Indonesia Diprediksi Kekurangan Air Bersih pada 2050. Artikel Lingkungan, GoodStats. (2025). https://data.goodstats.id/statistic/50-penduduk-indonesia-diprediksi-kekurangan-air-bersih-pada-2050-nNSet
