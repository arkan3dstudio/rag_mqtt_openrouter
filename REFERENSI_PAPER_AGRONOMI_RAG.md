# Referensi paper untuk penguatan sistem RAG agronomi realtime

## 1. Sensor NPK/pH/EC dan validasi agronomi

1. Sulaeman, Y., Sutanto, E., Kasno, A., Sunandar, N., & Purwaningrahayu, R. D. (2024). **Developing and Testing a Portable Soil Nutrient Detector in Irrigated and Rainfed Paddy Soils from Java, Indonesia**. *Computers, 13*(8), 209. DOI: 10.3390/computers13080209.
   - Relevansi: sangat dekat dengan sistem ini karena memakai sensor tanah portabel untuk padi Indonesia, mencakup suhu, moisture, pH, EC, N, P, dan K. Paper ini mendukung kebutuhan validasi sensor terhadap variasi tanah/lokasi sebelum memberi rekomendasi pupuk.

2. Reza, M. N. et al. (2025). **Trends of Soil and Solution Nutrient Sensing for Open Field Soil and Facilitated Hydroponic Cultivation**. *Sensors, 25*(2), 453.
   - Relevansi: dasar literatur untuk teknologi sensing hara tanah dan kebutuhan validasi sensor sebelum digunakan sebagai dasar keputusan agronomi.

3. Tornese, I. et al. (2024). **Use of Probes and Sensors in Agriculture—Current Trends and Future Perspectives**. *AgriEngineering, 6*(4), 234.
   - Relevansi: mendukung penggunaan sensor/probe dalam pertanian presisi, sekaligus menguatkan bahwa kalibrasi dan konteks lapang penting untuk interpretasi sensor.

4. Sobhy, D. M. et al. (2025). **Soil Nutrient Monitoring Technologies for Sustainable Agriculture**. *Sustainability, 17*(18), 8477.
   - Relevansi: review teknologi monitoring hara tanah untuk pertanian berkelanjutan dan dasar penguatan retrieval dokumen sensor/hara dalam RAG.

5. Chen, X., Zhang, H., & Wong, C. U. I. (2025). **Dynamic Monitoring and Precision Fertilization Decision System for Agricultural Soil Nutrients Using UAV Remote Sensing and GIS**. *Agriculture, 15*(15), 1627.
   - Relevansi: menunjukkan pentingnya integrasi data sensor, spasial, dan sistem keputusan pemupukan presisi. Cocok sebagai landasan bahwa rekomendasi pupuk perlu berbasis data spasial/lapang, bukan hanya satu pembacaan sensor.

## 2. LLM/RAG untuk pertanian

6. Zhu, H. et al. (2025). **Harnessing Large Vision and Language Models in Agriculture: A Review**. *Frontiers in Plant Science*.
   - Relevansi: menjelaskan bahwa RAG membantu LLM mengambil informasi dari knowledge base domain spesifik sehingga jawaban lebih grounded dan mengurangi risiko informasi usang/hallucination.

7. Wu, H. et al. (2025). **Crop GraphRAG: Pest and Disease Knowledge Base Q&A System for Sustainable Crop Protection**. *Frontiers in Plant Science*.
   - Relevansi: mendukung desain RAG domain pertanian dengan sumber pengetahuan terkurasi, prompt khusus domain, dan retrieval yang mempertimbangkan struktur pengetahuan.

8. Samuel, D. J., Skarga-Bandurova, I., Sikolia, D., & Awais, M. (2025). **AgroLLM: Connecting Farmers and Agricultural Practices through Large Language Models for Enhanced Knowledge Transfer and Practical Application**. arXiv:2503.04788.
   - Relevansi: menunjukkan pemanfaatan LLM + RAG untuk knowledge transfer pertanian dan evaluasi performa jawaban.

9. Yang, B. et al. (2025). **AgriGPT: a Large Language Model Ecosystem for Agriculture**. arXiv:2508.08632.
   - Relevansi: memperkuat argumen perlunya LLM domain pertanian, dataset terkurasi, dan multi-channel RAG untuk meningkatkan reasoning dan factual grounding.

10. Fanuel, M. et al. (2025). **AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice**. arXiv:2512.10114.
   - Relevansi: sangat penting untuk desain lanjutan karena rekomendasi agronomi harus region-aware. Untuk sistem ini, latitude/longitude, komoditas, fase, dan dokumen lokal sebaiknya masuk ke metadata filter/reranking.

## Implikasi desain untuk kode

- Result MQTT sebaiknya hanya dikirim saat retrieval RAG relevan dan LLM menghasilkan JSON valid jika mode strict diaktifkan.
- Jika RAG/LLM gagal, sistem tidak mengirim result ke topic result; sistem hanya mengirim feedback failed agar Kodular tidak membaca hasil fallback sebagai hasil RAG final.
- Untuk kalibrasi threshold, data sensor perlu dipasangkan dengan hasil uji lab dan label ahli/penyuluh.
- Threshold harus dipindah secara bertahap ke file konfigurasi eksternal berdasarkan crop, fase, jenis tanah, lokasi, status kalibrasi, jumlah sampel, dan error kalibrasi.
