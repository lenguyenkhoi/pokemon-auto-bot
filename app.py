import cv2

# Hàm xử lý khi click chuột trái: in ra toạ độ (x, y)
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Toạ độ đo được: (x={x}, y={y})")

# Đọc file ảnh giả lập
img = cv2.imread("screenshot.png")

# Cấu hình cửa sổ hiển thị
window_name = "Toa do"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, on_click)

# Hiển thị ảnh cho đến khi ấn phím bất kỳ
cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()