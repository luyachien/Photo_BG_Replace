import cv2
import numpy as np
import mediapipe as mp
import os
from rembg import remove as rembg_remove
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, colorchooser

# -----------------------------
# rembg 模組檢查
# -----------------------------
try:
    import rembg
    print("✅ rembg 已成功導入，版本:", rembg.__version__)
    rembg_available = True
except ImportError as e:
    rembg_available = False
    print("⚠️ 導入 rembg 失敗:", e)

# -----------------------------
# 去背函數
# -----------------------------
def remove_background(image_path, bg_color=(255, 255, 255), bg_image_path=None,
                      auto_refine=True, feather_radius=15, transparent=False):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_seg:
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ 找不到影像：{image_path}")
            return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = selfie_seg.process(rgb)
        raw_mask = np.clip(result.segmentation_mask, 0, 1)
        mean_conf = np.mean(raw_mask)

        # 動態閾值 & 增強遮罩
        raw_mask = raw_mask ** 0.8
        dynamic_thresh = np.mean(raw_mask)
        thresh = 0.45 + (dynamic_thresh - 0.5) * 0.2
        mask = (raw_mask > thresh).astype(np.float32)

        # 羽化邊緣
        if feather_radius > 0:
            mask = cv2.GaussianBlur(mask, (feather_radius*2+1, feather_radius*2+1), 0)
        alpha = np.clip(mask, 0.0, 1.0)

        # 背景處理
        if bg_image_path:
            bg_img = cv2.imread(bg_image_path)
            bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))
        else:
            bg_img = np.full(img.shape, bg_color, dtype=np.uint8)

        output = (img * alpha[:, :, None] + bg_img * (1 - alpha[:, :, None])).astype(np.uint8)

        # 使用 rembg 精修
        if auto_refine and rembg_available and mean_conf < 0.6:
            print("✨ Mediapipe 結果不夠乾淨，改用 Rembg 精修中...")
            _, buffer = cv2.imencode('.png', img)
            result_bytes = rembg_remove(buffer.tobytes())
            result_image = np.frombuffer(result_bytes, np.uint8)
            result_image = cv2.imdecode(result_image, cv2.IMREAD_UNCHANGED)

            if result_image.shape[2] == 4:
                alpha = result_image[:, :, 3] / 255.0
                if transparent:
                    output = result_image
                else:
                    output = (result_image[:, :, :3] * alpha[:, :, None] +
                              (bg_img.astype(np.float32) * (1 - alpha[:, :, None]))).astype(np.uint8)
            else:
                output = result_image

        # 透明背景輸出
        if transparent and output.shape[2] == 3:
            b, g, r = cv2.split(output)
            alpha_channel = (alpha * 255).astype(np.uint8)
            output = cv2.merge((b, g, r, alpha_channel))

        return output

# -----------------------------
# GUI 主程式
# -----------------------------
class BGReplaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智慧證件照背景替換")

        # 照片資料夾
        tk.Label(root, text="照片資料夾:").grid(row=0, column=0, sticky="w")
        self.input_dir_entry = tk.Entry(root, width=50)
        self.input_dir_entry.grid(row=0, column=1)
        tk.Button(root, text="瀏覽", command=self.select_input_dir).grid(row=0, column=2)

        # 背景顏色
        tk.Label(root, text="背景顏色:").grid(row=1, column=0, sticky="w")
        self.bg_color = (255, 255, 255)
        tk.Button(root, text="選擇顏色", command=self.choose_color).grid(row=1, column=1, sticky="w")

        # 顏色預覽方塊
        self.color_preview = tk.Label(root, bg=self.rgb_to_hex(self.bg_color),
                                      width=3, height=1, relief="groove", borderwidth=2)
        self.color_preview.grid(row=1, column=2, sticky="w")

        # 背景圖片
        tk.Label(root, text="背景圖片 (可選):").grid(row=2, column=0, sticky="w")
        self.bg_image_path = tk.StringVar()
        tk.Entry(root, textvariable=self.bg_image_path, width=50).grid(row=2, column=1)
        tk.Button(root, text="瀏覽", command=self.select_bg_image).grid(row=2, column=2)

        # 選項
        self.auto_refine_var = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="自動精修 (rembg)", variable=self.auto_refine_var).grid(row=3, column=0, sticky="w")
        self.transparent_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="輸出透明背景", variable=self.transparent_var).grid(row=3, column=1, sticky="w")

        # 開始按鈕
        tk.Button(root, text="開始處理", command=self.start_processing).grid(row=4, column=1)

        # 日誌面板
        self.log_text = scrolledtext.ScrolledText(root, width=70, height=15)
        self.log_text.grid(row=5, column=0, columnspan=3, pady=10)

    # -----------------------------
    # 輔助函數
    # -----------------------------
    def rgb_to_hex(self, rgb):
        """將 RGB 轉為 #RRGGBB 格式"""
        return "#%02x%02x%02x" % rgb

    # -----------------------------
    # 按鈕功能
    # -----------------------------
    def select_input_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_dir_entry.delete(0, tk.END)
            self.input_dir_entry.insert(0, folder)

    def choose_color(self):
        color = colorchooser.askcolor()[0]
        if color:
            self.bg_color = tuple(int(c) for c in color)
            self.color_preview.config(bg=self.rgb_to_hex(self.bg_color))

    def select_bg_image(self):
        file = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file:
            self.bg_image_path.set(file)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def start_processing(self):
        input_dir = self.input_dir_entry.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("錯誤", "請選擇有效的照片資料夾")
            return

        output_dir = os.path.join(input_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not files:
            messagebox.showwarning("提醒", "資料夾中沒有圖片")
            return

        for file in files:
            input_path = os.path.join(input_dir, file)
            self.log(f"🖼️ 處理中：{file}")
            result = remove_background(
                input_path,
                bg_color=self.bg_color,
                bg_image_path=self.bg_image_path.get() if self.bg_image_path.get() else None,
                auto_refine=self.auto_refine_var.get(),
                transparent=self.transparent_var.get()
            )
            if result is not None:
                base_name = os.path.splitext(file)[0]
                ext = ".png" if self.transparent_var.get() else ".jpg"

                # 序號另存新檔
                counter = 1
                while True:
                    output_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                    if not os.path.exists(output_path):
                        break
                    counter += 1

                cv2.imwrite(output_path, result)
                self.log(f"✅ 已儲存：{output_path}")

        messagebox.showinfo("完成", f"🎉 全部完成！結果在 '{output_dir}' 資料夾中。")

# -----------------------------
# 啟動 GUI
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BGReplaceApp(root)
    root.mainloop()
