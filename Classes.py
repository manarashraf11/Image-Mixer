import io
import logging
import numpy as np
from PyQt5.QtCore import QPoint, Qt, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QImage
from numpy.fft import fft2, fftshift
from PIL import Image
from PyQt5.QtWidgets import QFileDialog, QGraphicsView, QRubberBand, QWidget, QHBoxLayout, QSizeGrip

logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


class FileBrowser:
    def __init__(self, parent):
        self.parent = parent

    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self.parent, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Image Files (*.jpg *.jpeg *.png *.svg *.webp)",
                                                  options=options)
        if fileName:
            # Convert the image to grayscale
            img = self.read_file(fileName)
            img = img.convert('L')
            return img
        return None
        # if fileName:
        #     # Check if the image is grayscale
        #     is_grayscale = self.is_grayscale(fileName)
        #     # If not grayscale, convert it
        #     if not is_grayscale:
        #         self.convert_to_grayscale(fileName)
        #     return self.read_file(fileName)
        # else:
        #     return None, None

    def read_file(self, image_path):
        # Check the file extension to determine the format
        if image_path.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            # Use Pillow to open JPEG or PNG images
            img = Image.open(image_path)
            return img
        elif image_path.endswith('.svg'):
            # Use CairoSVG to convert SVG to PNG, then open with Pillow
            svg_data = open(image_path, 'rb').read()
            # png_data = cairosvg.svg2png(svg_data)
            img = Image.open(io.BytesIO(svg_data))
            return img
        else:
            print("Unsupported image format")
            return None


class Images:
    def __init__(self, img, label):
        self.label = label
        self.img_size = img.size
        # self.weights_lst = [0.0, 0.0, 0.0, 0.0]  # [mag, phase, real, imaginary]
        self.original_img = img  # Store the original image
        self.Pimg = None
        self.np_img = None
        self.Qimg = None
        self.update_image(img)
        self.ft = None
        self.Magnitude = None
        self.magnitude = None
        self.phase = None
        self.real = None
        self.imaginary = None
        self.compute_ft()
        # self.selected_region = None

    def update_image(self, img):
        self.img_size = img.size
        self.Pimg = img
        self.np_img = np.array(img)
        self.Qimg = self.pillow_to_qimage(img)
        return self

    def compute_ft(self):
        # Compute and store the FT, magnitude, phase, real, and imaginary components of the image.
        self.ft = fftshift(fft2(self.np_img))
        self.Magnitude = np.abs(self.ft)
        # ft_shifted = fftshift(fft2(self.np_img))
        self.magnitude = 20 * np.log(np.abs(self.ft))
        self.phase = np.angle(self.ft)
        self.real = np.real(self.ft)
        self.imaginary = np.imag(self.ft)

    def pillow_to_qimage(self, pillow_image):
        # Convert a Pillow Image to a QImage
        image_data = pillow_image.tobytes()
        bytes_per_line = pillow_image.width  # 1 byte per pixel for grayscale images
        qimage = QImage(image_data, pillow_image.width, pillow_image.height, bytes_per_line, QImage.Format_Grayscale8)
        return qimage

#
# class ClickableViewer(QGraphicsView):
#     rectangleDrawn = pyqtSignal(QRect)
#
#     def __init__(self, parent=None):
#         super(ClickableViewer, self).__init__(parent)
#         self.rect = None
#         self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
#         self.origin = QPoint()
#         self.dragging = False
#
#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             if self.rect and self.rect.contains(event.pos()):
#                 self.dragging = True
#             else:
#                 self.origin = QPoint(event.pos())
#                 self.rubberBand.setGeometry(QRect(self.origin, QSize()))
#                 self.rubberBand.show()
#
#     def mouseMoveEvent(self, event):
#         if not self.origin.isNull():
#             if self.dragging:
#                 diff = event.pos() - self.origin
#                 self.origin = event.pos()
#                 self.rect.translate(self.pos() + diff)
#                 self.rubberBand.setGeometry(self.rect)
#             else:
#                 self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
#
#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.dragging = False
#             self.rect = self.rubberBand.geometry()
#             print("Rectangle coordinates: ", self.rect)
#             print("Rect x: ", self.rect.x(), "Rect y: ", self.rect.y())
#             if self.scene() is None:
#                 print(f"There is no image to pick a region!")
#             else:
#                 self.rectangleDrawn.emit(self.rect)

                # # Get the first item in the scene
                # item = self.scene().items()[0]
                # logging.debug(f"item: {item}, type: {item.type()}")
                # pixmap = item.pixmap()
                # logging.debug(f"pixmap: {pixmap.isNull()}")
                # qimage = pixmap.toImage()
                # logging.debug(f"qimage: {qimage.isNull()}")
                # # Convert QImage to NumPy array
                # if not qimage.isNull():
                #     ptr = qimage.bits()
                #     ptr.setsize(qimage.byteCount())
                #     img_array = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)  # Copies the data
                #     # Extract the ROI from the NumPy array
                #     roi = img_array[self.rect.y():self.rect.y() + self.rect.height(),
                #           self.rect.x():self.rect.x() + self.rect.width()]
                #     # Now 'roi' contains the image rgba data in the ROI
                #     print(roi)
                # else:
                #     print("The QImage is null.")


class ResizableRubberBand(QWidget):
    rectangle_resized = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super(ResizableRubberBand, self).__init__(parent)
        self.setWindowFlags(Qt.SubWindow)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        grip1 = QSizeGrip(self)
        grip2 = QSizeGrip(self)
        layout.addWidget(grip1, 0, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(grip2, 0, Qt.AlignRight | Qt.AlignBottom)
        self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
        self.rubberband.move(0, 0)
        # self.rubberband.show()
        self.show()
        self.origin = None
        self.isMoving = False

    def resizeEvent(self, event):
        self.rubberband.resize(self.size())
        self.rubberband.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubberband.show()
            self.isMoving = True

    def mouseMoveEvent(self, event):
        if not self.origin.isNull() and self.isMoving:
            # self.move(event.pos() - self.origin)
            delta = event.pos() - self.origin
            self.move(self.pos() + delta)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = None
            self.isMoving = False
        self.rectangle_resized.emit(self.geometry())


class ClickableViewer(QGraphicsView):
    rectangleDrawn = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super(ClickableViewer, self).__init__(parent)
        self.rect = None
        # self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.rubberBand = ResizableRubberBand(self)
        # self.origin = QPoint()
        # self.isMoving = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = QPoint(event.pos())
            if not self.rubberBand.geometry().contains(self.origin):
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
            #     self.isMoving = False
            # else:
            #     self.isMoving = True

    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            # if self.isMoving:
            #     self.rubberBand.move(event.pos() - self.origin)
            # else:
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rect = self.rubberBand.geometry()
            print("Rectangle coordinates: ", self.rect)
            print("Rect x: ", self.rect.x(), "Rect y: ", self.rect.y())
            # self.rubberBand.hide()

            if self.scene() is None:
                print(f"There is no image to pick a region!")
            else:
                self.rectangleDrawn.emit(self.rect)

