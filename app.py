import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import io

from pencil_sketch import (
    custom_pencil_sketch,
    histogram_equalization_numpy,
    sobel_edge_detection_numpy,
    otsu_threshold_numpy,
    calculate_psnr_scratch,
    calculate_ssim_scratch,
    opencv_builtin_sketch_timed
)

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Hệ Thống Xử Lý Ảnh Thuần NumPy (From Scratch)", 
    page_icon="🎨",
    layout="wide"
)

# Header chính của ứng dụng
st.title("🎨 Ứng Dụng Xử Lý Ảnh Thuần NumPy (From Scratch)")
st.caption("Đồ án môn Xử lý ảnh: Cài đặt thuật toán thuần NumPy (Không dùng OpenCV.methods) & Xây dựng ứng dụng hoàn chỉnh")

# Sidebar cấu hình tham số
st.sidebar.header("⚙️ Cấu Hình Thuật Toán")

st.sidebar.markdown("### 1. Thuật toán Lọc Giữ Biên")
filter_mode = st.sidebar.radio(
    "Chọn bộ lọc làm mịn:",
    ["Guided Filter (Lọc định hướng giữ biên)", "Gaussian Filter (Separable 1D)"],
    index=0
)

if "Guided" in filter_mode:
    r_val = st.sidebar.slider("Bán kính cửa sổ (r)", min_value=2, max_value=32, value=16, step=2)
    eps_val = st.sidebar.select_slider(
        "Hệ số điều chuẩn (epsilon)",
        options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01
    )
    use_guided = True
    sigma_val = 4.0
else:
    sigma_val = st.sidebar.slider("Độ lệch chuẩn Gauss (sigma)", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
    r_val = 16
    eps_val = 0.01
    use_guided = False

st.sidebar.markdown("### 2. Tinh Chỉnh Nét Vẽ Chì (Tone Mapping)")
alpha_val = st.sidebar.slider("Độ tương phản nét chì (alpha)", 0.8, 1.6, 1.15, 0.05)
beta_val = st.sidebar.slider("Độ đậm nhạt than chì (beta)", -40, 20, -15, 5)

st.sidebar.markdown("### 3. Tham Số Đối Chứng OpenCV")
sigma_s_cv = st.sidebar.slider("OpenCV sigma_s (Không gian)", 10, 150, 60, 5)
sigma_r_cv = st.sidebar.slider("OpenCV sigma_r (Biên)", 0.01, 0.20, 0.07, 0.01)


# Hàm hỗ trợ tải ảnh PNG
def get_image_download_bytes(img_array):
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    else:
        pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# Hàm hỗ trợ xuất bảng số liệu Excel
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Benchmark_Result')
    return output.getvalue()


# Khu vực tải ảnh lên
uploaded_file = st.file_uploader(
    "📥 Tải lên ảnh của bạn (Chân dung, phong cảnh, ảnh y tế X-quang, chi tiết máy...)", 
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    # Đọc dữ liệu ảnh an toàn
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # Chuyển định dạng về BGR cho xử lý
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif len(img_np.shape) == 2:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Chuyển ảnh xám gốc bằng công thức chuẩn ITU-R BT.601
    gray_original = (0.299 * img_bgr[:, :, 2] + 0.587 * img_bgr[:, :, 1] + 0.114 * img_bgr[:, :, 0]).astype(np.uint8)

    # Hệ thống 3 Tabs chức năng
    tab1, tab2, tab3 = st.tabs([
        "🎨 1. Phác Thảo Chì (Custom vs OpenCV)", 
        "🏥 2. Thuật Toán Giáo Trình (Y Tế & Biên Sobel)", 
        "📊 3. Đánh Giá Định Lượng & Benchmark"
    ])

    # =========================================================================
    # TAB 1: PHÁC THẢO CHÌ (SO SÁNH CUSTOM NUMPY VS OPENCV)
    # =========================================================================
    with tab1:
        st.subheader("So Sánh Trực Quan: Thuật Toán Tự Code (Pure NumPy) vs Thư Viện OpenCV C++")

        # Chạy thuật toán tự code
        custom_sketch, t_custom = custom_pencil_sketch(
            img_bgr, 
            use_guided=use_guided, 
            sigma=sigma_val, 
            r=r_val, 
            eps=eps_val, 
            alpha=alpha_val, 
            beta=beta_val
        )

        # Chạy hàm mẫu của OpenCV để đối chứng
        cv_sketch, t_cv = opencv_builtin_sketch_timed(
            img_bgr, 
            sigma_s=sigma_s_cv, 
            sigma_r=sigma_r_cv
        )

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📷 Ảnh Gốc Đầu Vào")
            st.image(image, use_container_width=True)
            st.download_button(
                label="📥 Tải Ảnh Gốc",
                data=get_image_download_bytes(img_bgr),
                file_name="anh_goc.png",
                mime="image/png",
                use_container_width=True
            )

        with col2:
            st.markdown(f"#### 🛠️ Custom Pure NumPy ({t_custom:.2f} ms)")
            st.image(custom_sketch, channels="GRAY", use_container_width=True)
            st.caption(f"Phương pháp: {'Guided Filter (O(1) Integral)' if use_guided else 'Separable Gaussian'} + Custom Color Dodge")
            st.download_button(
                label="📥 Tải Ảnh Custom Sketch (NumPy)",
                data=get_image_download_bytes(custom_sketch),
                file_name="custom_sketch_numpy.png",
                mime="image/png",
                use_container_width=True
            )

        with col3:
            st.markdown(f"#### 📦 OpenCV Built-in ({t_cv:.2f} ms)")
            st.image(cv_sketch, channels="GRAY", use_container_width=True)
            st.caption("Phương pháp: cv2.pencilSketch (C++ Domain Transform Filter)")
            st.download_button(
                label="📥 Tải Ảnh OpenCV Sketch",
                data=get_image_download_bytes(cv_sketch),
                file_name="opencv_sketch_builtin.png",
                mime="image/png",
                use_container_width=True
            )

    # =========================================================================
    # TAB 2: THUẬT TOÁN GIÁO TRÌNH (HISTOGRAM EQUALIZATION & SOBEL FROM SCRATCH)
    # =========================================================================
    with tab2:
        st.subheader("Cài Đặt Thuật Toán Giáo Trình Xử Lý Ảnh Kinh Điển (From Scratch)")
        
        col_med, col_ind = st.columns(2)

        with col_med:
            st.markdown("### 🏥 Module 1: Cân Bằng Lược Đồ Mức Xám (Histogram Equalization)")
            st.write("Thuật toán giáo trình trải rộng dải động mức xám bằng hàm phân phối tích lũy xác suất (CDF) thuần NumPy. Chuyên dụng làm nổi bật chi tiết tổn thương mô/xương trên ảnh Y tế (X-quang, CT).")
            
            eq_img, hist_orig, hist_eq = histogram_equalization_numpy(img_bgr)
            st.image(eq_img, caption="Kết quả Cân bằng Histogram (From Scratch)", channels="GRAY", use_container_width=True)
            
            # Hiển thị số liệu phân bố
            hist_df = pd.DataFrame({
                "Mức xám (0-255)": list(range(256)),
                "Ảnh Gốc": hist_orig,
                "Sau Cân Bằng": hist_eq
            })
            st.line_chart(hist_df.set_index("Mức xám (0-255)"))
            
            st.download_button(
                label="📥 Tải Ảnh Tăng Cường (Histogram Equalized)",
                data=get_image_download_bytes(eq_img),
                file_name="histogram_equalized.png",
                mime="image/png",
                use_container_width=True
            )

        with col_ind:
            st.markdown("### 🏭 Module 2: Phát Hiện Biên Sobel & Phân Ngưỡng Otsu")
            st.write("Sử dụng toán tử đạo hàm không gian Sobel 3x3 và thuật toán phân ngưỡng tự động Otsu cực đại hóa phương sai liên lớp (Between-class variance) để trích xuất biên dạng cấu trúc cơ khí/công nghiệp.")
            
            sobel_thresh = st.slider("Ngưỡng lọc biên Sobel:", 10, 150, 45, 5)
            sobel_mag, sobel_bin = sobel_edge_detection_numpy(img_bgr, threshold=sobel_thresh)
            otsu_seg, best_t = otsu_threshold_numpy(gray_original)
            
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                st.image(sobel_mag, caption="Độ lớn Gradient Sobel", channels="GRAY", use_container_width=True)
            with sub_c2:
                st.image(otsu_seg, caption=f"Phân ngưỡng Otsu (Ngưỡng tối ưu T*={best_t})", channels="GRAY", use_container_width=True)
                
            st.download_button(
                label="📥 Tải Bản Đồ Biên Sobel",
                data=get_image_download_bytes(sobel_mag),
                file_name="sobel_edges.png",
                mime="image/png",
                use_container_width=True
            )

    # =========================================================================
    # TAB 3: ĐÁNH GIÁ ĐỊNH LƯỢNG & BENCHMARK
    # =========================================================================
    with tab3:
        st.subheader("📊 Bảng Đánh Giá Định Lượng Hiệu Năng & Độ Bảo Toàn Cấu Trúc")
        st.write("So sánh khoa học giữa phương pháp Tự code (Pure NumPy) và Thư viện OpenCV C++ bằng các chỉ số đo lường tín hiệu và độ tương đồng cấu trúc.")

        # Tính toán chỉ số from scratch
        psnr_custom = calculate_psnr_scratch(gray_original, custom_sketch)
        ssim_custom = calculate_ssim_scratch(gray_original, custom_sketch)
        
        psnr_cv = calculate_psnr_scratch(gray_original, cv_sketch)
        ssim_cv = calculate_ssim_scratch(gray_original, cv_sketch)

        df_metrics = pd.DataFrame({
            "Chỉ số đánh giá": [
                "Thời gian xử lý (ms)", 
                "PSNR (dB - Càng cao sai số càng nhỏ)", 
                "SSIM (0 - 1.0, Đo độ bảo toàn cấu trúc)",
                "Cơ chế thuật toán",
                "Tính độc lập kích thước cửa sổ"
            ],
            "Thuật toán Tự code (NumPy From Scratch)": [
                f"{t_custom:.2f} ms",
                f"{psnr_custom:.2f} dB",
                f"{ssim_custom:.4f}",
                "Guided Filter O(1) + Color Dodge Phi tuyến",
                "Độc lập O(1) nhờ Ảnh Tích Lũy (Integral Image)"
            ],
            "Thư viện OpenCV (cv2.pencilSketch)": [
                f"{t_cv:.2f} ms",
                f"{psnr_cv:.2f} dB",
                f"{ssim_cv:.4f}",
                "Domain Transform Recursive Filter (Black-box)",
                "Phụ thuộc tham số không gian/bán kính"
            ]
        })

        st.table(df_metrics.set_index("Chỉ số đánh giá"))

        st.markdown("#### 📁 Xuất Dữ Liệu Benchmark:")
        col_csv, col_xlsx = st.columns(2)
        
        csv_bytes = df_metrics.to_csv(index=False).encode('utf-8-sig')
        with col_csv:
            st.download_button(
                label="📄 Tải Bảng Kết Quả (File .CSV)",
                data=csv_bytes,
                file_name="benchmark_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        xlsx_bytes = convert_df_to_excel(df_metrics)
        with col_xlsx:
            st.download_button(
                label="📊 Tải Bảng Kết Quả (File .XLSX)",
                data=xlsx_bytes,
                file_name="benchmark_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.success("""
        **Nhận xét phân tích phục vụ bảo vệ đồ án:**
        1. **Thời gian thực thi:** Nhờ kỹ thuật **Ảnh Tích Lũy (Integral Image)** được vector hóa 100% bằng NumPy, thuật toán tự code đạt tốc độ xử lý theo thời gian thực (vài chục mili-giây) mà không cần gọi các thư viện C++ biên dịch sẵn.
        2. **Chỉ số SSIM & PSNR:** Bộ lọc Guided Filter giữ vững các đường biên cạnh mạnh và chi tiết sắc nét của chủ thể, đồng thời loại bỏ nhiễu phẳng mịn, mang lại chỉ số tương đồng cấu trúc cao và nét vẽ chì rõ ràng hơn.
        """)

else:
    st.info("👆 Vui lòng tải một hình ảnh lên ở thanh bên trên để bắt đầu chạy ứng dụng!")
