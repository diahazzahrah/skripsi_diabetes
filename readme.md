SKRIPSI
- DIAH SITI FATIMAH AZZAHRAH
- NIM. 4611419056
- PROGRAM STUDI TEKNIK INFORMATIKA 2019

DIATEC (Diabetes Detection) Application 
Aplikasi ini dapat mengklasifikasikan penyakit diabetes dengan "non diabetes" atau "diabetes" sesuai dengan inputan angka yang dilakukan user. Sistem ini dirancang untuk menguji skripsi yang disusun dengan judul "Komparasi Algoritma Probabilistic Neural Network (PNN) dan k-Nearest Neighbor (k-NN) Untuk Klasifikasi Penyakit Diabetes". Selain itu, pada sistem ini, terdapat tampilan akurasi, dataset, home, dan about. Sistem ini menggunakan bantuan teknologi Artificial Intelligence yaitu Data Science (Data mining). Akan tetapi, sistem ini belum dapat diimplementasikan pada bidang kesehatan secara langsung karena kesalahannya masih lebih besar dari 0.5%. 

Cara Menjalankan Project dari Skripsi_Diabetes.
1. Pastikan laptop/PC yang digunakan sudah menginstall python v.3 
(pada proses pembuatan ini menggunakan python v 3.8.0)
2. Extract terlebih dahulu project yang berformat rar. (skripsi_diabetes)
3. Lalu, buka project folder skripsi_diabetes melalui visual studio code 
4. Pada project ini, sudah terdapat environment yang dinamai .venv 
5. Lalu, coba aktifkan environment tersebut. 
6. Apabila tidak bisa diaktifkan, coba melakukan penghapusan folder .venv, dan membuat environment yang baru. 
7. Setelah environment baru telah dibuat, dan tetap masih belum dapat diaktifkan, maka cobalah mengikuti instruksi pada link ini (https://www.stanleyulili.com/powershell/solution-to-running-scripts-is-disabled-on-this-system-error-on-powershell/) 
8. Jika .venv sudah aktif, maka install library yang dibutuhkan (secara manual)
     - pip install Flask 
     - pip install numpy (versi 1.20.3)
     - pip install neupy 
     - pip install pandas
     - pip install scikit-learn 
9. Cara untuk mengupgrade library numpy pada sistem dengan menggunakan 'pip3 install --upgrade numpy==1.20.3'
10. Check version pip. (must 23.0.1) -> apabila versi belum 23.0.1, maka lakukan upgrade. 
11. Execute the command/terminal:
     `python app.py`