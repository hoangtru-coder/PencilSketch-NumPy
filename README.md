# 🎨 PencilSketch-NumPy: Hệ Thống Xử Lý Ảnh Thuần NumPy (From Scratch)

> **Đồ án môn học: Xử Lý Ảnh Số (Digital Image Processing)**  
> Cài đặt toàn bộ thuật toán lõi bằng toán đại số ma trận NumPy — Hoàn toàn không sử dụng các hàm lọc đóng gói sẵn của OpenCV.

---

## 🌟 Điểm Sáng Kỹ Thuật

- **Pure NumPy 100% (From Scratch):** Toàn bộ các thuật toán xử lý ảnh, trích xuất biên, cân bằng lược đồ và đo lường (PSNR, SSIM) được tự cài đặt từ con số 0.
- **Guided Image Filter $O(1)$ (IEEE TPAMI 2013):** Khai thác cấu trúc dữ liệu **Ảnh tích lũy 2 chiều (Integral Image / Summed-Area Table)** giúp thời gian lọc đạt độ phức tạp $O(1)$ bất biến theo bán kính cửa sổ lọc $r$ (~34.9 ms cho ảnh $512 \times 512$).
- **Bảo toàn cấu trúc vượt trội:** Chỉ số tương đồng cấu trúc SSIM cao hơn từ **3.6 lần đến 17.2 lần** so với hàm thư viện `cv2.pencilSketch` của OpenCV.
- **Ứng dụng đa miền (3 Tabs Streamlit):**
  1. *Tab 1 (Nghệ thuật số):* Chuyển ảnh phong cảnh/chân dung thành tranh phác thảo chì than (Color Dodge + Charcoal Tone Mapping).
  2. *Tab 2 (Y sinh & Công nghiệp):* Cân bằng lược đồ mức xám (Histogram Equalization CDF) cho ảnh X-quang/CT và Tách biên Sobel 3×3 + Phân đoạn Otsu tự động cho chi tiết cơ khí.
  3. *Tab 3 (Benchmark):* Đối chứng định lượng thời gian (ms), PSNR (dB), SSIM và xuất báo cáo CSV / Excel (.XLSX).

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Phần Mềm

### 1. Cài đặt các thư viện phụ thuộc
Mở terminal tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy giao diện Web tương tác
```bash
python -m streamlit run app.py
```
Trình duyệt sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`.

---

## 📂 Cấu Trúc Mã Nguồn

```text
├── app.py                  # Giao diện Web tương tác Streamlit (3 Tabs, xuất PNG, CSV, XLSX)
├── pencil_sketch.py        # Thư viện thuật toán xử lý ảnh số lõi thuần NumPy (From Scratch)
├── requirements.txt        # Danh mục thư viện phụ thuộc (streamlit, numpy, opencv, pillow, pandas, openpyxl)
├── BAO_CAO_PROJECT.md      # Báo cáo nghiệm thu đồ án chi tiết (Markdown chuẩn học thuật)
├── BAO_CAO_PROJECT.docx    # Báo cáo nghiệm thu định dạng Microsoft Word hoàn chỉnh
├── .gitignore              # Bộ lọc bỏ file tạm cache của Python
└── README.md               # Tài liệu giới thiệu dự án
```

---

## 📖 Báo Cáo Nghiệm Thu
Chi tiết về cơ sở toán học, phân tích độ phức tạp thuật toán, khảo sát thực nghiệm và phân tích các trường hợp thất bại (Failure Cases) được trình bày đầy đủ tại:
- [Báo cáo Markdown (BAO_CAO_PROJECT.md)](BAO_CAO_PROJECT.md)
- [Báo cáo Microsoft Word (BAO_CAO_PROJECT.docx)](BAO_CAO_PROJECT.docx)

---

## 📚 Tài Liệu Tham Khảo Chính
1. Kaiming He, Jian Sun, Xiaoou Tang, *"Guided Image Filtering"*, IEEE TPAMI, Vol. 35, No. 6, 2013.
2. Zhou Wang et al., *"Image Quality Assessment: From Error Visibility to Structural Similarity"*, IEEE TIP, 2004.
3. Rafael C. Gonzalez, Richard E. Woods, *"Digital Image Processing"*, 4th Edition, Pearson Education.
