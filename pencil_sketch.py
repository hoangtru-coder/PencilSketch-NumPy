"""
=============================================================================
MODULE XỬ LÝ ẢNH THUẦN NUMPY (FROM SCRATCH) CHO ĐỒ ÁN MÔN XỬ LÝ ẢNH
=============================================================================
Đặc tả kỹ thuật:
- Toàn bộ thuật toán lõi được cài đặt FROM SCRATCH bằng toán ma trận NumPy.
- KHÔNG sử dụng các hàm xử lý ảnh có sẵn của OpenCV (như cv2.boxFilter, 
  cv2.GaussianBlur, cv2.bilateralFilter, cv2.createCLAHE, cv2.Canny, cv2.divide).
- OpenCV chỉ được dùng để nạp/ghi ảnh (I/O) và đổi hệ màu ban đầu.
- Tích hợp bài báo nghiên cứu khoa học: "Guided Image Filtering" 
  (Kaiming He, Jian Sun, Xiaoou Tang - IEEE TPAMI 2013 / ECCV 2010).
=============================================================================
"""

import numpy as np
import time
import cv2


# =============================================================================
# 1. CÁC THUẬT TOÁN LỌC KHÔNG GIAN CƠ BẢN (SPATIAL FILTERING FROM SCRATCH)
# =============================================================================

def box_filter_numpy(I, r):
    """
    Lọc hộp (Box Filter) tính trung bình vùng lân cận (2r+1)x(2r+1)
    Thuật toán O(1) mỗi pixel sử dụng Ảnh Tích Lũy 2D (Integral Image / Cumulative Sum).
    Cài đặt 100% bằng NumPy, hoàn toàn không dùng cv2.boxFilter.
    """
    I = I.astype(np.float32)
    H, W = I.shape
    k = 2 * r + 1
    
    # Mở rộng biên (padding edge) để xử lý pixel sát cạnh
    padded = np.pad(I, ((r + 1, r), (r + 1, r)), mode='edge')
    
    # Tính ảnh tích lũy 2 chiều
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    
    # Tọa độ 4 góc của cửa sổ trượt
    y2 = np.arange(k, k + H)
    x2 = np.arange(k, k + W)
    y1 = np.arange(0, H)
    x1 = np.arange(0, W)
    
    # Tính tổng nhanh bằng công thức hình chữ nhật: D - B - C + A
    window_sum = (
        integral[y2[:, None], x2]
        - integral[y1[:, None], x2]
        - integral[y2[:, None], x1]
        + integral[y1[:, None], x1]
    )
    return window_sum / float(k * k)


def gaussian_kernel_1d(sigma, radius=None):
    """
    Tự sinh kernel Gauss 1D theo công thức giải tích chuẩn giáo trình:
    G(x) = exp(-x^2 / (2 * sigma^2)) / sum(G)
    """
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def gaussian_filter_numpy(image, sigma=3.0):
    """
    Bộ lọc Gauss 2D tách chiều (Separable 2D Gaussian Filter) thuần NumPy.
    Giảm độ phức tạp từ O(K^2) xuống O(2K) bằng cách tích chập theo hàng rồi theo cột.
    Không dùng cv2.GaussianBlur.
    """
    img_float = image.astype(np.float32)
    kernel_1d = gaussian_kernel_1d(sigma)
    r = len(kernel_1d) // 2
    
    # 1. Tích chập theo phương ngang (Row-wise convolution)
    pad_h = np.pad(img_float, ((0, 0), (r, r)), mode='edge')
    conv_h = np.zeros_like(img_float, dtype=np.float32)
    for idx, w in enumerate(kernel_1d):
        conv_h += w * pad_h[:, idx:idx + img_float.shape[1]]
        
    # 2. Tích chập theo phương dọc (Column-wise convolution)
    pad_v = np.pad(conv_h, ((r, r), (0, 0)), mode='edge')
    conv_v = np.zeros_like(img_float, dtype=np.float32)
    for idx, w in enumerate(kernel_1d):
        conv_v += w * pad_v[idx:idx + img_float.shape[0], :]
        
    return conv_v


# =============================================================================
# 2. BỘ LỌC ĐỊNH HƯỚNG GIỮ BIÊN (GUIDED FILTER - IEEE TPAMI / ECCV) FROM SCRATCH
# =============================================================================

def guided_filter_numpy(I, p, r=16, eps=0.01):
    """
    Bộ lọc định hướng Guided Image Filter (Kaiming He, ECCV 2010 / IEEE TPAMI 2013).
    Cài đặt 100% bằng NumPy, sử dụng thuật toán Integral Image O(1) độc lập kích thước cửa sổ.
    
    Tham số:
    - I: Ảnh hướng dẫn (Guidance image, dải [0, 1])
    - p: Ảnh đầu vào cần lọc (Filtering input, dải [0, 1])
    - r: Bán kính cửa sổ cục bộ
    - eps: Hệ số điều chuẩn regularization (tránh chia cho 0 và kiểm soát độ mịn biên)
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    
    # 1. Tính giá trị trung bình cục bộ bằng box_filter_numpy
    mean_I = box_filter_numpy(I, r)
    mean_p = box_filter_numpy(p, r)
    corr_I = box_filter_numpy(I * I, r)
    corr_Ip = box_filter_numpy(I * p, r)
    
    # 2. Tính hiệp phương sai và phương sai cục bộ
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    # 3. Nghiệm giải tích mô hình hồi quy tuyến tính cục bộ q = a * I + b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # 4. Trung bình trượt của các hệ số a và b
    mean_a = box_filter_numpy(a, r)
    mean_b = box_filter_numpy(b, r)
    
    # 5. Tái tạo ảnh lọc đầu ra q
    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)


# =============================================================================
# 3. THUẬT TOÁN PHÁC THẢO CHÌ TỰ CODE (CUSTOM PENCIL SKETCH FROM SCRATCH)
# =============================================================================

def color_dodge_numpy(front, back):
    """
    Phép hòa trộn Color Dodge chuyển ảnh thành nét vẽ chì thuần NumPy.
    Công thức: Dodge(A, B) = min(255, (A * 256) / (255 - B + eps))
    Không dùng cv2.divide hay cv2.convertScaleAbs.
    """
    front = front.astype(np.float32)
    back = back.astype(np.float32)
    
    # Tránh chia cho 0 bằng giá trị epsilon nhỏ
    denominator = 255.0 - back
    denominator = np.where(denominator == 0, 1e-4, denominator)
    
    result = (front * 256.0) / denominator
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def adjust_contrast_brightness_numpy(image, alpha=1.15, beta=-15):
    """
    Cân chỉnh độ tương phản và độ sâu nét chì thuần NumPy:
    O(x, y) = clip(alpha * I(x, y) + beta, 0, 255)
    """
    adjusted = alpha * image.astype(np.float32) + beta
    return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


def custom_pencil_sketch(img_bgr, use_guided=True, sigma=4.0, r=16, eps=0.01, alpha=1.15, beta=-15):
    """
    Quy trình phác thảo tranh vẽ chì toàn diện (From Scratch bằng NumPy):
    1. Chuyển đổi mức xám (Grayscale).
    2. Lọc làm mịn giữ biên (Edge-preserving Smoothing) bằng Guided Filter hoặc Gaussian Filter.
    3. Đảo ngược âm bản (Inversion).
    4. Làm mờ ảnh âm bản (Negative Blur).
    5. Hòa trộn nét chì Color Dodge phi tuyến.
    6. Cân chỉnh độ tương phản than chì (Charcoal Tone Mapping).
    """
    start_time = time.time()
    
    # 1. Chuyển sang ảnh xám bằng trọng số chuẩn ITU-R BT.601
    if len(img_bgr.shape) == 3:
        b = img_bgr[:, :, 0].astype(np.float32)
        g = img_bgr[:, :, 1].astype(np.float32)
        r_ch = img_bgr[:, :, 2].astype(np.float32)
        gray = 0.299 * r_ch + 0.587 * g + 0.114 * b
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        gray = img_bgr.copy().astype(np.uint8)

    # 2. Làm mịn ảnh giữ biên
    if use_guided:
        I = gray.astype(np.float32) / 255.0
        smoothed_float = guided_filter_numpy(I, I, r=r, eps=eps)
        smoothed = (smoothed_float * 255.0).astype(np.uint8)
    else:
        smoothed = np.clip(gaussian_filter_numpy(gray, sigma=sigma), 0, 255).astype(np.uint8)

    # 3. Tạo ảnh âm bản
    inverted = 255 - smoothed

    # 4. Làm mờ ảnh âm bản bằng Gaussian tách chiều thuần NumPy
    # Dùng sigma lớn để tạo hiệu ứng bóng đổ nét chì mềm mại
    blur_sigma = max(3.0, sigma * 2.5)
    blurred_inv = np.clip(gaussian_filter_numpy(inverted, sigma=blur_sigma), 0, 255).astype(np.uint8)

    # 5. Phép hòa trộn nét chì Color Dodge thuần NumPy
    sketch = color_dodge_numpy(gray, blurred_inv)

    # 6. Tăng cường độ đậm nhạt than chì
    final_sketch = adjust_contrast_brightness_numpy(sketch, alpha=alpha, beta=beta)
    
    elapsed_ms = (time.time() - start_time) * 1000.0
    return final_sketch, elapsed_ms


# =============================================================================
# 4. CÁC THUẬT TOÁN GIÁO TRÌNH KINH ĐIỂN CÀI ĐẶT FROM SCRATCH
# =============================================================================

def histogram_equalization_numpy(image):
    """
    Cân bằng lược đồ mức xám (Histogram Equalization) from scratch thuần NumPy.
    Ứng dụng tăng cường độ tương phản cho ảnh Y tế (X-quang, MRI) và ảnh thiếu sáng.
    Thay thế hoàn toàn cho cv2.equalizeHist và cv2.createCLAHE.
    """
    if len(image.shape) == 3:
        gray = (0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]).astype(np.uint8)
    else:
        gray = image.copy().astype(np.uint8)
        
    H, W = gray.shape
    total_pixels = H * W
    
    # 1. Tính tần suất các mức xám [0..255]
    hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
    
    # 2. Tính hàm phân phối tích lũy xác suất (Cumulative Distribution Function - CDF)
    cdf = hist.cumsum()
    
    # 3. Chuẩn hóa CDF theo công thức giáo trình DIP:
    # h(v) = round((cdf(v) - cdf_min) / (total_pixels - cdf_min) * 255)
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    
    if cdf_max > cdf_min:
        cdf_normalized = (cdf_masked - cdf_min) * 255.0 / (cdf_max - cdf_min)
    else:
        cdf_normalized = cdf_masked * 0
        
    lookup_table = np.ma.filled(cdf_normalized, 0).astype(np.uint8)
    
    # 4. Ánh xạ lại toàn bộ ảnh theo lookup table
    equalized_img = lookup_table[gray]
    
    # Tính lại histogram của ảnh sau khi cân bằng để đối chiếu
    hist_eq, _ = np.histogram(equalized_img.flatten(), 256, [0, 256])
    
    return equalized_img, hist, hist_eq


def sobel_edge_detection_numpy(image, threshold=50):
    """
    Phát hiện biên Sobel (Sobel Edge Operator) from scratch thuần NumPy.
    - Tự tạo kernel đạo hàm không gian Kx, Ky.
    - Tính tích chập 2D.
    - Tính biên độ gradient G = sqrt(Gx^2 + Gy^2).
    - Phân ngưỡng nhị phân để lọc biên rõ nét.
    """
    if len(image.shape) == 3:
        gray = (0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]).astype(np.float32)
    else:
        gray = image.astype(np.float32)
        
    # Tiền lọc mượt nhẹ để giảm nhiễu trước khi tính đạo hàm
    smoothed = gaussian_filter_numpy(gray, sigma=1.0)
    
    # Định nghĩa toán tử Sobel chuẩn
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
                   
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)
                   
    H, W = smoothed.shape
    padded = np.pad(smoothed, ((1, 1), (1, 1)), mode='edge')
    
    Gx = np.zeros((H, W), dtype=np.float32)
    Gy = np.zeros((H, W), dtype=np.float32)
    
    for i in range(3):
        for j in range(3):
            Gx += Kx[i, j] * padded[i:i + H, j:j + W]
            Gy += Ky[i, j] * padded[i:i + H, j:j + W]
            
    # Tính độ lớn Gradient
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude_norm = (magnitude / max_val * 255.0).astype(np.uint8)
    else:
        magnitude_norm = np.zeros((H, W), dtype=np.uint8)
        
    # Nhị phân hóa biên theo ngưỡng
    binary_edges = np.where(magnitude_norm >= threshold, 255, 0).astype(np.uint8)
    
    return magnitude_norm, binary_edges


def otsu_threshold_numpy(gray_image):
    """
    Phân ngưỡng tự động Otsu (Otsu's Thresholding) from scratch thuần NumPy.
    Tối đa hóa phương sai giữa các lớp (Between-class variance) để tìm ngưỡng phân đoạn tối ưu.
    """
    hist, _ = np.histogram(gray_image.flatten(), 256, [0, 256])
    total_pixels = gray_image.size
    
    sum_total = np.dot(np.arange(256), hist)
    weight_bg = 0
    sum_bg = 0
    max_variance = 0.0
    best_threshold = 0
    
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break
            
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        
        # Phương sai giữa các lớp: sigma_B^2 = w_bg * w_fg * (mu_bg - mu_fg)^2
        var_between = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
        
        if var_between > max_variance:
            max_variance = var_between
            best_threshold = t
            
    segmented = np.where(gray_image >= best_threshold, 255, 0).astype(np.uint8)
    return segmented, best_threshold


# =============================================================================
# 5. CHỈ SỐ ĐÁNH GIÁ ĐỊNH LƯỢNG (PSNR & SSIM FROM SCRATCH)
# =============================================================================

def calculate_psnr_scratch(img1, img2):
    """
    Tính chỉ số Tỷ lệ Tín hiệu trên Nhiễu Cực đại (PSNR) from scratch:
    PSNR = 10 * log10(MAX_I^2 / MSE)
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim_scratch(img1, img2):
    """
    Tính chỉ số Độ tương đồng Cấu trúc (SSIM - Wang et al., 2004) from scratch bằng NumPy.
    SSIM(x, y) = [ (2*mu_x*mu_y + C1) * (2*sigma_xy + C2) ] / [ (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2) ]
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    
    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2
    
    # Tính kỳ vọng cục bộ (local means) bằng box filter
    r = 5
    mu_x = box_filter_numpy(x, r)
    mu_y = box_filter_numpy(y, r)
    
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = box_filter_numpy(x * x, r) - mu_x_sq
    sigma_y_sq = box_filter_numpy(y * y, r) - mu_y_sq
    sigma_xy = box_filter_numpy(x * y, r) - mu_xy
    
    numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


# =============================================================================
# 6. THƯ VIỆN ĐỐI CHỨNG OPENCV C++ BUILT-IN (DÙNG ĐỂ BENCHMARK KHOA HỌC)
# =============================================================================

def opencv_builtin_sketch_timed(img_np, sigma_s=60, sigma_r=0.07):
    """
    Hàm mẫu có sẵn của OpenCV (Black-box C++) dùng làm thước đo đối chứng với Custom Code.
    """
    start_time = time.time()
    _, cv_res = cv2.pencilSketch(img_np, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=0.05)
    if len(cv_res.shape) == 3:
        cv_res_gray = cv2.cvtColor(cv_res, cv2.COLOR_BGR2GRAY)
    else:
        cv_res_gray = cv_res
    elapsed = (time.time() - start_time) * 1000.0
    return cv_res_gray, elapsed
