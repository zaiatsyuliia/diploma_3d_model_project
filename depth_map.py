import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import filters

class DepthMap:
    def __init__(self, model_type="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.feature_extractor = None

    def load_model(self):
        try:
            if self.model_type == "DPT_Large":
                try:
                    from transformers import DPTForDepthEstimation, DPTFeatureExtractor
                    self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
                    self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
                    self.model.to(self.device).eval()
                    print("DPT модель завантажена")
                    return
                except:
                    pass
            
            torch.hub.set_dir('./models')
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            self.model.to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms.dpt_transform if "DPT" in self.model_type else transforms.small_transform
            print(f"Модель {self.model_type} завантажена")
            
        except Exception as e:
            print(f"Використовуємо простий алгоритм: {e}")
            self.model = "simple"

    def simple_depth_estimation(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape
        
        # Градієнти та краї
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(grad_x**2 + grad_y**2)
        edges = edges / (edges.max() + 1e-8)
        
        # Фокус
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        focus = np.abs(gray.astype(float) - blur.astype(float))
        focus = focus / (focus.max() + 1e-8)
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - w//2)**2 + (y - h//2)**2)
        position_bias = 1 - (distance / np.sqrt((w//2)**2 + (h//2)**2))
        
        # Вертикальний градієнт
        vertical = np.linspace(1, 0.3, h).reshape(-1, 1)
        vertical = np.tile(vertical, (1, w))
        
        depth = edges * 0.4 + focus * 0.3 + position_bias * 0.2 + vertical * 0.1
        return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    def enhance_depth(self, depth, img):
        """Комплексне покращення карти глибини"""
        # Підвищення контрасту
        depth_enhanced = cv2.equalizeHist((depth * 255).astype(np.uint8)) / 255.0
        
        # Гауссове згладжування
        depth_smooth = filters.gaussian(depth_enhanced, sigma=1.0)
        
        # Додавання країв
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0 if len(img.shape) == 3 else img / 255.0
        edges = filters.sobel(gray)
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        depth_combined = depth_smooth + 0.3 * edges
        
        # Гамма корекція
        depth_gamma = np.power(depth_combined, 0.7)
        
        # Посилення центру
        h, w = depth_gamma.shape
        y, x = np.ogrid[:h, :w]
        dist_center = np.sqrt((x - w//2)**2 + (y - h//2)**2)
        center_mask = 1 - (dist_center / np.sqrt((w//2)**2 + (h//2)**2))
        depth_final = depth_gamma * (1 + 0.5 * center_mask)
        
        # Сегментація об'єкта
        try:
            gray_int = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            binary = cv2.adaptiveThreshold(gray_int, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel), 
                                    cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                mask = np.zeros(gray_int.shape, np.uint8)
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.fillPoly(mask, [largest_contour], 255)
                mask_norm = mask / 255.0
                depth_final = np.where(mask > 0, depth_final * mask_norm + 0.3, depth_final * 0.1)
        except:
            pass  # Якщо сегментація не вдалася, продовжуємо без неї
        
        # Нормалізація
        return (depth_final - depth_final.min()) / (depth_final.max() - depth_final.min() + 1e-8)

    def estimate_depth(self, img):
        """Оцінка глибини"""
        if self.model == "simple" or self.model is None:
            depth = self.simple_depth_estimation(img)
        else:
            try:
                if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
                    inputs = self.feature_extractor(images=img_rgb, return_tensors="pt")
                    with torch.no_grad():
                        depth = self.model(**inputs).predicted_depth.squeeze().cpu().numpy()
                    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
                    input_batch = self.transform(img_rgb).to(self.device)
                    with torch.no_grad():
                        prediction = self.model(input_batch)
                        depth = F.interpolate(prediction.unsqueeze(1), size=img_rgb.shape[:2],
                                            mode="bicubic", align_corners=False).squeeze().cpu().numpy()
                
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            except Exception as e:
                print(f"Помилка з моделлю: {e}. Використовуємо простий алгоритм")
                depth = self.simple_depth_estimation(img)
        
        return self.enhance_depth(depth, img)

    def show_depth(self, original, depth, model_type):
        """Візуалізація результатів"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if len(original.shape) == 3 else original, 
                  cmap='gray' if len(original.shape) == 2 else None)
        plt.title("Оригінал")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(depth, cmap='plasma')
        plt.title(f"Карта глибини ({model_type})")
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"depth_map_{model_type}.png", bbox_inches='tight', dpi=150)
        print(f"Збережено як depth_map_{model_type}.png") 
        
def main():
    img_path = "cat.jpg"
    models = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]

    img = cv2.imread(img_path)
    if img is None:
        print("Не знайдено зображення, створюю випадкове...")
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite("test.jpg", img)
        img = cv2.imread("test.jpg")

    # Обробка всіх моделей
    for model_type in models:
        try:
            depth_estimator = DepthMap(model_type)
            depth_estimator.load_model()
            depth_map = depth_estimator.estimate_depth(img)
            depth_estimator.show_depth(img, depth_map, model_type)
        except Exception as e:
            print(f"Помилка з моделлю {model_type}: {e}")
            depth_estimator = DepthMap("simple")
            depth_estimator.model = "simple"
            depth_map = depth_estimator.estimate_depth(img)
            depth_estimator.show_depth(img, depth_map, f"{model_type}_fallback")

    plt.show()

if __name__ == "__main__":
    main()