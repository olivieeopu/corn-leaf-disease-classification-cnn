# corn-leaf-disease-classification-cnn
Klasifikasi empat kelas kondisi daun jagung menggunakan CNN berbasis AlexNet dengan TensorFlow/Keras, mencakup EDA citra, preprocessing, modifikasi arsitektur, dan evaluasi per kelas.

# Klasifikasi Penyakit Daun Jagung Menggunakan CNN

Proyek akademik computer vision untuk mengklasifikasikan gambar daun jagung ke dalam empat kelas menggunakan Convolutional Neural Network (CNN) berbasis AlexNet.

Proyek mencakup eksplorasi dataset citra, preprocessing, implementasi arsitektur secara manual tanpa pretrained weights, modifikasi model, serta evaluasi performa klasifikasi.

Model modifikasi memperoleh **accuracy 89,21%** dan **macro ROC-AUC 0,9847** pada 630 gambar test. Namun, performanya belum merata pada seluruh kelas, terutama Gray Leaf Spot.

## Tujuan

- Mengeksplorasi distribusi kelas, ukuran gambar, aspect ratio, dan intensitas piksel.
- Mengimplementasikan baseline CNN berbasis AlexNet.
- Membandingkan baseline dengan konfigurasi modifikasi.
- Mengevaluasi hasil menggunakan accuracy, precision, recall, F1-score, confusion matrix, dan ROC-AUC.
- Mengidentifikasi kelas yang masih sulit dibedakan.

## Dataset

Dataset yang digunakan dalam tugas akademik berisi **4.188 gambar** dengan empat kelas:

| Label | Kelas | Jumlah gambar |
|---|---|---:|
| 0 | Blight | 1.146 |
| 1 | Common Rust | 1.306 |
| 2 | Gray Leaf Spot | 574 |
| 3 | Healthy | 1.162 |
| **Total** | | **4.188** |

Gray Leaf Spot memiliki jumlah gambar paling sedikit sehingga distribusi kelas tidak seimbang.

Sumber asli dan lisensi dataset belum dicantumkan dalam laporan. Informasi tersebut perlu dilengkapi sebelum dataset didistribusikan melalui repository.

<img width="812" height="270" alt="Screenshot 2026-09-28 at 18 32 34" src="https://github.com/user-attachments/assets/369ceec1-54ab-4e49-9970-e87823c74caf" />

## Tools

- Python
- TensorFlow/Keras
- scikit-learn
- NumPy dan pandas
- Matplotlib
- Pillow
- Google Colab

## Alur Pengerjaan

### 1. Exploratory Data Analysis

Eksplorasi mencakup:

- Menghitung jumlah gambar per kelas.
- Menampilkan contoh gambar.
- Memeriksa ukuran, aspect ratio, dan intensitas piksel.

Pemeriksaan ukuran dan intensitas menggunakan sampel hingga 50 gambar per kelas, sehingga hasilnya tidak mewakili pemeriksaan menyeluruh terhadap seluruh file.

### 2. Pembagian Data

Gambar dibagi per kelas menggunakan `train_test_split` dengan `random_state=42`, mengikuti proporsi sekitar 70:15:15.

| Subset | Jumlah gambar |
|---|---:|
| Training | 2.930 |
| Validation | 628 |
| Test | 630 |

Laporan juga memuat generator awal dengan pembagian 70:30 untuk eksplorasi. Pelatihan akhir menggunakan folder train, validation, dan test yang dibentuk secara terpisah.

### 3. Preprocessing

- Mengubah ukuran gambar menjadi **224 × 224 piksel** dengan tiga channel.
- Menormalisasi nilai piksel ke rentang **0–1** menggunakan `rescale=1./255`.
- Menggunakan label kategorikal untuk multiclass classification.
- Menggunakan batch size **32**.
- Mengatur `shuffle=False` pada test generator agar urutan prediksi sesuai dengan label.

Data augmentation belum diterapkan pada konfigurasi yang ditampilkan.

## Model

### Baseline CNN Berbasis AlexNet

Arsitektur diimplementasikan secara manual menggunakan TensorFlow/Keras, tanpa pretrained weights.

```text
Input: 224 × 224 × 3
↓
Conv2D: 96 filters, 11 × 11, stride 4, ReLU
MaxPooling
↓
Conv2D: 256 filters, 5 × 5, ReLU
MaxPooling
↓
Conv2D: 384 filters, 3 × 3, ReLU
Conv2D: 384 filters, 3 × 3, ReLU
Conv2D: 256 filters, 3 × 3, ReLU
MaxPooling
↓
Flatten
Dense: 4096, ReLU
Dropout: 0.5
Dense: 4096, ReLU
Dropout: 0.5
↓
Dense: 4, Softmax

Baseline menggunakan optimizer Adam dan categorical cross-entropy, serta dilatih selama 10 epoch tanpa early stopping.



### Model Modifikasi

Perubahan yang diterapkan:

- Menambahkan **Batch Normalization setelah convolutional layer pertama dan kedua**.
- Mengurangi ukuran dense layers menjadi **2048 dan 1024 neuron**.
- Mempertahankan satu **Dropout 0.5** setelah dense layer pertama.
- Mengatur learning rate Adam menjadi **0.0001**.
- Menggunakan **class weighting** berdasarkan distribusi training set.
- Menerapkan **EarlyStopping** dengan `patience=5` dan `restore_best_weights=True`.

Baseline sudah menggunakan dropout. Jadi, modifikasi tidak memperkenalkan dropout untuk pertama kalinya, melainkan mengubah konfigurasi yang digunakan.

Model modifikasi dijadwalkan untuk maksimum 20 epoch dan berhenti setelah 16 epoch. Log menunjukkan validation loss terendah pada epoch ke-11.

## Hasil Evaluasi

Hasil berikut berasal dari output evaluasi model modifikasi pada **test set berisi 630 gambar**.

| Metrik | Nilai |
|---|---:|
| Accuracy | **89,21%** |
| Macro precision | **0,88** |
| Macro recall | **0,85** |
| Macro F1-score | **0,86** |
| Macro ROC-AUC, one-vs-rest | **0,9847** |

Precision, recall, dan F1-score mengikuti pembulatan dua desimal pada classification report.

### Performa per Kelas

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Blight | 0,81 | 0,87 | 0,84 | 172 |
| Common Rust | 0,93 | 0,97 | 0,95 | 196 |
| Gray Leaf Spot | 0,80 | 0,56 | 0,66 | 87 |
| Healthy | 0,97 | 0,99 | 0,98 | 175 |

### Confusion Matrix

<img width="505" height="431" alt="Screenshot 2026-09-28 at 18 31 59" src="https://github.com/user-attachments/assets/36c10ef9-015b-48be-af76-4439b72f904b" />

Urutan kelas:

- **0:** Blight
- **1:** Common Rust
- **2:** Gray Leaf Spot
- **3:** Healthy

Sebanyak **49 dari 87** gambar Gray Leaf Spot diklasifikasikan dengan benar, sementara **30 gambar** diprediksi sebagai Blight.

Hal ini menunjukkan bahwa kemampuan model membedakan Gray Leaf Spot dan Blight masih perlu ditingkatkan.

### Kurva Training

### Baseline

<img width="552" height="401" alt="Screenshot 2026-09-28 at 18 29 56" src="https://github.com/user-attachments/assets/e56eab51-1aaf-42bf-9aa3-045f76aa078b" />

<img width="542" height="385" alt="Screenshot 2026-09-28 at 18 30 17" src="https://github.com/user-attachments/assets/8fa962fa-dff5-45c1-b7b3-64850e74c101" />

### Tuned
<img width="806" height="491" alt="Screenshot 2026-09-28 at 18 31 09" src="https://github.com/user-attachments/assets/633d35c5-d232-4501-9ae7-daf0c5f13216" />

<img width="806" height="489" alt="Screenshot 2026-09-28 at 18 31 27" src="https://github.com/user-attachments/assets/09fc7fcf-01e9-4d95-a986-8ad0441d2ac9" />


Training accuracy meningkat, tetapi validation accuracy dan validation loss masih berfluktuasi. Early stopping digunakan untuk mengembalikan bobot dari epoch dengan validation loss terbaik.

## Temuan Utama

1. **Performa model modifikasi belum merata antarkelas.**  
   Healthy dan Common Rust memiliki recall tinggi, sedangkan Gray Leaf Spot masih memiliki recall sekitar 56%.

2. **Accuracy keseluruhan perlu dibaca bersama metrik per kelas.**  
   Accuracy 89,21% tidak berarti seluruh jenis penyakit dikenali dengan tingkat keberhasilan yang sama.

3. **ROC-AUC tinggi tidak menghilangkan kesalahan klasifikasi.**  
   ROC-AUC mengevaluasi pemeringkatan skor probabilitas, sedangkan confusion matrix menunjukkan keputusan kelas berdasarkan prediksi akhir.

4. **Beberapa komponen diubah sekaligus.**  
   Perubahan arsitektur, learning rate, class weighting, dan pengaturan training dilakukan bersamaan. Eksperimen ini belum mengisolasi kontribusi setiap perubahan.


## Dataset
https://drive.google.com/drive/folders/1C73TQXpWOW-MTnRBEIfQNsLWPDp7iQZW?usp=sharing
