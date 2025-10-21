import cv2
import numpy as np

class QRNavigationSystem:
    def __init__(self):
        self.roberts_kernel = np.array([[1, 0], [0, -1]])
        self.shadowing_kernel = np.array([
            [-0.1, 0.1, -0.1],
            [0.1, 0.5, 0.1],
            [-0.1, 0.1, -0.1],
        ])
    
    def decode_qr_code(self, image_path):
        """CV-2-15: Детектирование и декодирование QR-кода с помощью OpenCV"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка: не удалось загрузить изображение QR-кода: {image_path}")
            return None, None
        
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(image)
        
        if bbox is not None and data:
            points = []
            for point in bbox[0]:
                x, y = int(point[0]), int(point[1])
                points.append((x, y))
            
            # Визуализация QR-кода
            for i in range(4):
                cv2.line(image, points[i], points[(i + 1) % 4], (255, 0, 0), 3)
            
            cv2.putText(image, f"QR: {data}", (points[0][0], points[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Сохраняем изображение с выделенным QR-кодом
            qr_output_path = "qr_detected.jpg"
            cv2.imwrite(qr_output_path, image)
            print(f"QR-код обработан и сохранен как: {qr_output_path}")
            
            return data, points
        
        print("QR-код не обнаружен на изображении")
        return None, None
    
    def detect_road_marking(self, image_path):
        """CV-2-37: Обнаружение дорожной разметки"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Ошибка: не удалось загрузить изображение разметки: {image_path}")
            return None, None
        
        # Затемнение изображения
        frame_filtered = cv2.filter2D(frame, -1, self.shadowing_kernel)
        
        # Перевод в серый и фильтрация
        gray = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray, -1, self.roberts_kernel)
        
        # Детекция краев
        edges = cv2.Canny(img_out, 150, 200)
        
        # Поиск линий
        lines = cv2.HoughLinesP(edges, rho=1.5, theta=np.pi/180, threshold=50,
                               minLineLength=20, maxLineGap=20)
        
        # Визуализация разметки
        output_image = frame.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Сохраняем изображение с выделенной разметкой
        marking_output_path = "marking_detected.jpg"
        cv2.imwrite(marking_output_path, output_image)
        print(f"Разметка обработана и сохранена как: {marking_output_path}")
        
        return lines, output_image.shape
    
    def analyze_direction(self, lines, image_shape):
        """Анализ направления на основе дорожной разметки"""
        if lines is None:
            return "stop"  # Нет разметки - остановка
        
        # Анализ углов и положения линий
        left_lines = []
        right_lines = []
        center_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Вычисление угла наклона
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            else:
                angle = 90
                
            # Классификация линий
            if angle < -10:  # Левая линия (отрицательный угол)
                left_lines.append(line)
            elif angle > 10:  # Правая линия (положительный угол)
                right_lines.append(line)
            else:  # Центральная линия (почти горизонтальная)
                center_lines.append(line)
        
        # Логика определения направления
        if len(center_lines) > 2:
            return "forward"
        elif len(left_lines) > len(right_lines):
            return "right"
        elif len(right_lines) > len(left_lines):
            return "left"
        elif len(left_lines) == len(right_lines) and len(left_lines) > 0:
            return "forward"
        else:
            return "stop"
    
    def determine_final_command(self, qr_command, marking_direction):
        """Определение итоговой команды на основе QR и разметки"""
        # Базовые команды из QR-кода
        qr_commands = {
            "вперёд": "forward", "forward": "forward",
            "налево": "left", "left": "left", 
            "направо": "right", "right": "right",
            "стоп": "stop", "stop": "stop"
        }
        
        qr_dir = qr_commands.get(qr_command.lower(), "forward")
        
        # Приоритет: безопасность (разметка имеет высший приоритет)
        if marking_direction == "stop":
            return "STOP"  # Аварийная остановка
        elif marking_direction == "forward":
            # Если разметка позволяет движение вперед, выполняем команду из QR
            return qr_dir.upper()
        else:
            # Если разметка указывает поворот, корректируем команду
            if qr_dir == "forward":
                return marking_direction.upper()
            else:
                # Компромисс: выполняем команду из QR, но с осторожностью
                return f"{qr_dir.upper()} (CAUTION)"
    
    def process_navigation(self, qr_image_path, marking_image_path):
        """Основная функция обработки навигации"""
        print("=" * 50)
        print("СИСТЕМА НАВИГАЦИИ ПО QR-КОДАМ")
        print("=" * 50)
        
        # 1. Детектирование QR-кода
        print("1. Обработка QR-кода...")
        qr_data, qr_points = self.decode_qr_code(qr_image_path)
        qr_command = qr_data if qr_data else "вперёд"  # Команда по умолчанию
        
        print(f"   QR-команда: {qr_command}")
        
        # 2. Обнаружение дорожной разметки
        print("2. Обработка дорожной разметки...")
        lines, image_shape = self.detect_road_marking(marking_image_path)
        
        if lines is not None:
            print(f"   Обнаружено линий разметки: {len(lines)}")
        else:
            print("   Линии разметки не обнаружены")
        
        # 3. Определение направления по разметке
        print("3. Анализ направления движения...")
        marking_direction = self.analyze_direction(lines, image_shape)
        print(f"   Направление по разметке: {marking_direction}")
        
        # 4. Определение итоговой команды
        print("4. Формирование итоговой команды...")
        final_command = self.determine_final_command(qr_command, marking_direction)
        print(f"   Итоговая команда: {final_command}")
        
        # Создаем итоговое изображение с информацией
        result_image = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(result_image, f"QR COMMAND: {qr_command}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result_image, f"ROAD MARKING: {marking_direction}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result_image, f"FINAL COMMAND: {final_command}", (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Сохранение итогового результата
        output_path = "navigation_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"5. Итоговый результат сохранен как: {output_path}")
        
        return final_command, result_image

def main():
    navigation_system = QRNavigationSystem()
    
    # Запрос путей к отдельным изображениям
    print("Введите пути к изображениям:")
    qr_path = input("Путь к изображению с QR-кодом: ")
    marking_path = input("Путь к изображению с дорожной разметкой: ")
    
    result = navigation_system.process_navigation(qr_path, marking_path)
    
    if result:
        final_command, result_image = result
        print("\n" + "=" * 50)
        print(f"=== КОМАНДА ДЛЯ РОБОТА: {final_command} ===")
        print("=" * 50)

if __name__ == "__main__":
    main()