from ultralytics import YOLO
from PIL import Image
import cv2
import os
import random
from PyQt5.QtWidgets import QWidget, QGraphicsScene, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from tas_kagit_makas import Ui_Form
import numpy as np

class_names = ["Kağıt", "Taş", "Makas"]
model = YOLO("rock_paper_scissors.pt")

class TasKagitMakas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.taskagitmakas = Ui_Form()
        self.taskagitmakas.setupUi(self)
        self.taskagitmakas.oyna.clicked.connect(self.Oyna)
        
        self.scene1 = QGraphicsScene(self)
        self.taskagitmakas.aigoruntu.setScene(self.scene1)
        
        self.timer = QTimer(self)
        self.scene2 = QGraphicsScene(self)
        self.taskagitmakas.kullanicigoruntu.setScene(self.scene2)
        
        self.webcam = cv2.VideoCapture(0)
        self.webcam_active = True
        self.update_webcam()
        self.ai_skor = 0
        self.kullanici_skor = 0

    def Oyna(self):
        self.restart_webcam()
        self.countdown_value = 3
        self.showCountdown()

    def showCountdown(self):
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowTitle("Geri Sayım")
    
        font = self.msg_box.font()
        font.setPointSize(14)
        self.msg_box.setFont(font)
    
        self.updateCountdownMessage()
    
        self.msg_box.setStyleSheet("QMessageBox { min-width: 300px; }")
    
        self.msg_box.show()
    
        self.timer.singleShot(1000, self.updateCountdown) 

    def updateCountdown(self):
        self.countdown_value -= 1
        if self.countdown_value > 0:
            self.updateCountdownMessage()
            self.timer.singleShot(1000, self.updateCountdown)
        else:
            self.msg_box.accept()
            self.show_rastgele_foto_and_capture_webcam()

    def updateCountdownMessage(self):
        self.msg_box.setText(f"Hazırlan : {self.countdown_value}")

    def show_rastgele_foto_and_capture_webcam(self):
        self.gosterRastgeleFoto()
        self.webcam_active = False
        QTimer.singleShot(0, self.yakalaKullaniciGoruntusu)  

    def gosterRastgeleFoto(self):
        klasor_yolu = 'taskagitmakas'
        fotoğraflar = [f for f in os.listdir(klasor_yolu) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not fotoğraflar:
            return
        
        rastgele_foto = random.choice(fotoğraflar)
        fotoğraf_yolu = os.path.join(klasor_yolu, rastgele_foto)
        
        annotated_image = self.tahminYap(fotoğraf_yolu)
        if annotated_image is not None:
            h, w, ch = annotated_image.shape
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            q_image = QImage(annotated_image.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            view_size = self.taskagitmakas.aigoruntu.size()
            pixmap = pixmap.scaled(view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.scene1.clear()
            self.scene1.addPixmap(pixmap)
            self.update_skor()

    def yakalaKullaniciGoruntusu(self):
        if not self.webcam_active:
            return
        
        ret, frame = self.webcam.read()
        if not ret:
            return
        
        annotated_frame = self.tahminYapWebcam(frame)
        if annotated_frame is not None:
            h, w, ch = annotated_frame.shape
            q_image = QImage(annotated_frame.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            
            view_size = self.taskagitmakas.kullanicigoruntu.size()
            pixmap = pixmap.scaled(view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.scene2.clear()
            self.scene2.addPixmap(pixmap)
            self.update_skor()

    def update_webcam(self):
        if not self.webcam_active:
            return
        
        ret, frame = self.webcam.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame, 1)  
            annotated_frame = self.tahminYapWebcam(frame_rgb)
            if annotated_frame is not None:
                h, w, ch = annotated_frame.shape
                q_image = QImage(annotated_frame.data, w, h, ch * w, QImage.Format_RGB888)
                
                pixmap = QPixmap.fromImage(q_image)
                
                view_size = self.taskagitmakas.kullanicigoruntu.size()
                pixmap = pixmap.scaled(view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                self.scene2.clear()
                self.scene2.addPixmap(pixmap)
                
        QTimer.singleShot(30, self.update_webcam)

    def restart_webcam(self):
        self.webcam_active = True
        self.update_webcam()

    def tahminYap(self, image_path):
        im1 = Image.open(image_path)
        results = model.predict(source=im1, save=False, device='cuda')
        
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                class_id = int(classes[i])
                class_name = class_names[class_id]
                label = f'{class_name}'
                self.taskagitmakas.ai.setText(label)
        
  
        
        im1_np = np.array(im1)
        if im1_np.shape[2] == 4:  
            im1_np = cv2.cvtColor(im1_np, cv2.COLOR_RGBA2RGB)
        im1_bgr = cv2.cvtColor(im1_np, cv2.COLOR_RGB2BGR)
        return im1_bgr

  

    def tahminYapWebcam(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im1 = Image.fromarray(frame_rgb)
        results = model.predict(source=im1, save=False, device='cuda')
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].astype(int)
                class_id = int(classes[i])
                class_name = class_names[class_id] 
                label = f'{class_name}'

                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                self.taskagitmakas.kullanici.setText(label)
        
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        return frame
    
    def update_skor(self):
        ai_tercih = self.taskagitmakas.ai.text()
        kullanici_tercih = self.taskagitmakas.kullanici.text()

        if ai_tercih == kullanici_tercih:
            kazanan = "Berabere"
        
        else:

            if (ai_tercih == "Taş" and kullanici_tercih == "Makas") or \
                (ai_tercih == "Makas" and kullanici_tercih == "Kağıt") or \
                (ai_tercih == "Kağıt" and kullanici_tercih == "Taş"):
                self.ai_skor += 1
                self.taskagitmakas.aiskor.display(self.ai_skor)
                kazanan = "AI"
            else:
                self.kullanici_skor += 1
                self.taskagitmakas.kullaniciskor.display(self.kullanici_skor)
                kazanan = "Kullanıcı"

        self.show_round_winner(kazanan)
        self.check_game_over()


    def show_round_winner(self, kazanan):
        if kazanan == "Berabere":
            message = "Bu tur berabere!"
        else:
            message = f"Turun Kazananı: {kazanan}\nSkor: {self.ai_skor} - {self.kullanici_skor}"

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Tur Sonucu")
        msg_box.setText(message)
        msg_box.setStyleSheet("QMessageBox { min-width: 300px; }")
        msg_box.exec_()


    def check_game_over(self):
        if self.ai_skor == 3 or self.kullanici_skor == 3:
            kazanan = "AI" if self.ai_skor == 3 else "Kullanıcı"
            self.show_game_over_message(kazanan)
            self.reset_game()

    def show_game_over_message(self, kazanan):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Oyun Bitti!")
        msg_box.setText(f"Oyun Bitti! Kazanan: {kazanan}")
        msg_box.exec_()

    def reset_game(self):
        self.ai_skor = 0
        self.kullanici_skor = 0
        self.taskagitmakas.aiskor.display(self.ai_skor)
        self.taskagitmakas.kullaniciskor.display(self.kullanici_skor)
        self.scene1.clear()
        self.scene2.clear()
        self.taskagitmakas.ai.clear()
        self.taskagitmakas.kullanici.clear()


    def __del__(self):
        if hasattr(self, 'webcam'):
            self.webcam.release()
            cv2.destroyAllWindows()