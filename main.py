import traceback
import logging
import threading
import numpy as np
import qimage2ndarray
from functools import partial
from PIL import ImageEnhance
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal, QObject
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QRubberBand
from scipy.ndimage import zoom
import pyqtgraph as pg
import threading
# from PyQt5.QtCore import pyqtSignal, QObject
import cv2
from Classes import FileBrowser, Images, ClickableViewer, ResizableRubberBand
import task4UI
import sys
import warnings

try:
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))


    warnings.showwarning = warn_with_traceback

    # Set up logging
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(levelname)s - %(message)s')
    logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)


    class WorkerSignals(QObject):
        progress = pyqtSignal(int)


    class Worker(threading.Thread):
        def __init__(self, function, args, kwargs):
            super(Worker, self).__init__()
            self.function = function
            self.args = args
            self.kwargs = kwargs
            self.signals = WorkerSignals()

        def run(self):
            result = self.function(*self.args, **self.kwargs)
            self.signals.progress.emit(result)


    class ClickableLabel(QtWidgets.QLabel):
        def __init__(self, parent=None):
            super(ClickableLabel, self).__init__(parent)
            self.is_dragging = False
            self.prev_pos = None
            # Brightness and contrast of the original image
            self.brightness = 1.0
            self.contrast = 1.0

        # right double click to reset contrast\brightness !!!!!!!!!!!!!!!!!!!!!!

        def mousePressEvent(self, event):
            self.is_dragging = True
            self.prev_pos = event.pos()

        def mouseReleaseEvent(self, event):
            self.is_dragging = False
            # Clear the previous position
            self.prev_pos = None

        def mouseMoveEvent(self, event):
            if self.is_dragging:
                if self.prev_pos is not None:
                    try:
                        dx = event.x() - self.prev_pos.x()
                        dy = event.y() - self.prev_pos.y()
                        dx /= 1 * self.width()  # Increase this value to decrease sensitivity
                        dy /= 1 * self.height()  # Increase this value to decrease sensitivity
                        logging.debug(f"dx: {dx}, dy: {dy}")
                        if abs(dx) > abs(dy):
                            # If primarily horizontal, only change brightness
                            self.brightness += dx
                            dy = 0  # Ignore vertical movement
                        else:
                            # If primarily vertical, only change contrast
                            self.contrast -= dy
                            dx = 0  # Ignore horizontal movement
                        self.brightness = min(max(self.brightness, 0.5), 20)
                        self.contrast = min(max(self.contrast, 0.5), 20)
                        self.window().change_brightness(self)
                        self.window().change_contrast(self)
                    except Exception as e:
                        logging.error(f"mouseMoveEvent: {e}", exc_info=True)
                self.prev_pos = event.pos()

        # def mousePressEvent(self, event):
        #     self.is_dragging = True
        #     self.start_pos = event.pos()

        # def mouseReleaseEvent(self, event):
        #     self.is_dragging = False
        #     # Clear the start position
        #     self.start_pos = None

        # def mouseMoveEvent(self, event):
        #     if self.is_dragging:
        #         if self.prev_pos is not None:
        #             try:
        #                 dx = event.x() - self.prev_pos.x()
        #                 dy = event.y() - self.prev_pos.y()
        #                 # Scale dx and dy to a suitable range (-0.1 to 0.1)
        #                 dx /= 10 * self.width()
        #                 dy /= 10 * self.height()
        #                 logging.debug(f"dx: {dx}, dy: {dy}")
        #                 # Define a dead zone for mouse movement (adjust this value as needed)
        #                 dead_zone = 0.01
        #                 # If movement in x is within the dead zone, consider it as no horizontal movement
        #                 if abs(dx) < dead_zone:
        #                     dx = 0
        #                 # If movement in y is within the dead zone, consider it as no vertical movement
        #                 if abs(dy) < dead_zone:
        #                     dy = 0
        #                 # Add dx and dy to the current brightness and contrast
        #                 self.brightness += dx
        #                 self.contrast -= dy
        #                 # Ensure brightness and contrast are within a certain range (0.5 to 1.5)
        #                 self.brightness = min(max(self.brightness, 0.5), 1.5)
        #                 self.contrast = min(max(self.contrast, 0.5), 1.5)
        #                 # Use the new brightness and contrast values
        #                 self.window().change_brightness(self)
        #                 self.window().change_contrast(self)
        #             except Exception as e:
        #                 logging.error(f"mouseMoveEvent: {e}", exc_info=True)
        #         self.prev_pos = event.pos()

        # def mouseDoubleClickEvent(self, event):
        #     self.parent().open_image(self)


    class MainApp(QtWidgets.QMainWindow, task4UI.Ui_MainWindow):
        def __init__(self):
            super(MainApp, self).__init__()
            self.setupUi(self)
            self.fileBrowser = FileBrowser(self)
            # Lists
            self.labels = [self.label_img1, self.label_img2, self.label_img3, self.label_img4]
            self.ft_labels = [self.label_img1_fft, self.label_img2_fft, self.label_img3_fft, self.label_img4_fft]
            self.comboBoxes = [self.comboBox_img1, self.comboBox_img2, self.comboBox_img3, self.comboBox_img4]
            self.radioButtons_moods = [self.radioButton_Mag_phase, self.radioButton_Real_Imaginary]
            self.comboxItems = ["Magnitude", "Phase"]
            self.used_indices = []
            self.roi_rect = None
            self.qft_img = None
            self.radioButton_Mag_phase.setChecked(True)
            self.smallest_size = None
            self.min_width = float('inf')
            self.min_height = float('inf')

            self.weights_lst = [0.0, 0.0, 0.0, 0.0]  # [w1, w2, w3, w4]
            self.weighted_sums = {}
            for key in ['Magnitude', 'phase', 'real', 'imaginary']:
                self.weighted_sums[key] = np.zeros(([]), dtype='complex_')
            # Dictionaries
            self.lbl_images_dic = {}  # Dictionary to store Image instances
            self.img_indices_lbls_dic = {i + 1: label for i, label in enumerate(self.labels)}
            self.components_indices_dic = {'Magnitude': 0,
                                           'Phase': 1,
                                           'Real': 2,
                                           'Imaginary': 3}
            self.img_ft_labels_dic = {ft_label: img_label for ft_label, img_label in zip(self.ft_labels, self.labels)}
            self.scenes = {label: QGraphicsScene() for label in self.img_ft_labels_dic.keys()}
            self.combo_boxes = {ft_label: combo_box for ft_label, combo_box in zip(self.ft_labels, self.comboBoxes)}
            self.images_roi = {}
            # Connecting
            # Connect double-click signal to open_image method
            for label in self.labels:
                label.mouseDoubleClickEvent = lambda event, lbl=label: self.open_image(lbl)

            def make_lambda(ft_label):
                return lambda: self.display_ft_component(ft_label)

            for ft_label, combo_box in zip(self.ft_labels, self.comboBoxes):
                combo_box.activated.connect(make_lambda(ft_label))

            for i in range(1, 5):
                slider = getattr(self, f"horizontalSlider_img{i}")
                combo_box = getattr(self, f"comboBox_img{i}")
                slider.setRange(0, 100)
                slider.setValue(0)
                slider.setSingleStep(1)
                slider.valueChanged.connect(lambda value, index=i: self.update_weights_lst(value, index))
                combo_box.activated.connect(lambda: self.components_mixer())

            for viewer in self.ft_labels:
                viewer.rectangleDrawn.connect(self.draw_rectangles)  # Signal & Slot
                viewer.rubberBand.rectangle_resized.connect(self.draw_rectangles)  # Signal & Slot

            self.radioButton_Mag_phase.toggled.connect(self.update_mood)

            self.comboBox.activated.connect(self.calc_roi)

        def update_mood(self):
            # sender = self.sender()
            # if sender.text() == "Magnitude - Phase" and sender.isChecked():
            if self.radioButton_Mag_phase.isChecked():
                self.comboxItems = ["Magnitude", "Phase"]
            else:
                self.radioButton_Real_Imaginary.setChecked(True)
                self.comboxItems = ["Real", "Imaginary"]

            for comboBox in self.comboBoxes:
                comboBox.clear()
                comboBox.addItems(self.comboxItems)
            for ft_label in self.ft_labels:
                self.scenes[ft_label].clear()

        def draw_rectangles(self, rect):
            self.roi_rect = 5
            # self.comboBox.setCurrentText('Inner region')
            logging.debug(f"draw_rectangles, rect: {rect}")
            self.calc_roi(rect)
            # Draw the ROI on all ft components
            for viewer in self.ft_labels:
                if viewer.scene() is not None:
                    viewer.rubberBand.setGeometry(rect)
                    # viewer.rubberBand = ResizableRubberBand(viewer)
                    # viewer.rubberBand.update()
                    viewer.rubberBand.show()

        def open_image(self, label):
            try:
                # logging.debug(f"min_width = {self.min_width}, min_height = {self.min_height}")
                # Open a file dialog to select an image
                img = self.fileBrowser.browse_file()
                if img is not None:
                    # Create an Image instance with the associated label
                    image_instance = Images(img, label)
                    # Store the Image instance in the dictionary
                    self.lbl_images_dic[label] = image_instance
                    logging.debug(f"lbl_images_dic: {self.lbl_images_dic.items()}")
                    # self.img_weights[image_instance] = self.initial_weights_lst
                    # If there is only one image, this image has the smallest size
                    if len(self.lbl_images_dic) == 1:
                        self.min_width = image_instance.img_size[0]
                        self.min_height = image_instance.img_size[1]
                        # logging.debug(f"min_width = {self.min_width}, min_height = {self.min_height}")
                    else:
                        # Resize all images to the smallest one
                        image_instance = self.resize_images(image_instance)
                        # logging.debug(f"min_width = {self.min_width}, min_height = {self.min_height}")
                    # Log the value of qimg
                    # logging.debug(f"Qimg to the func= {image_instance.Qimg}")
                    # logging.debug(f"img_instance = {image_instance}")
                    # Process all images and set them to their labels
                    for label, img in self.lbl_images_dic.items():
                        pixmap = self.process_image_to_show(img.Qimg)
                        label.setPixmap(pixmap)
            except Exception as e:
                logging.error(f"Error1: {e}")
                tb1 = traceback.format_exc()
                logging.error(tb1)

        def change_brightness(self, label):
            try:
                brightness = label.brightness
                logging.debug(f"Changing brightness to: {brightness}")
                self.change_image_property(label, ImageEnhance.Brightness, brightness)
            except Exception as e:
                logging.error(f"change_Brightness: {e}", exc_info=True)

        def change_contrast(self, label):
            try:
                contrast = label.contrast
                self.change_image_property(label, ImageEnhance.Contrast, contrast)
            except Exception as e:
                logging.error(f"change_Contrast: {e}", exc_info=True)

        def change_image_property(self, label, enhancer_class, value):
            try:
                logging.debug(f"enhancer_class: {enhancer_class}, value: {value}")
                image_instance = self.lbl_images_dic[label]
                enhancer = enhancer_class(image_instance.original_img)  # Use the original image here
                enhanced_img = enhancer.enhance(value)
                image_instance.update_image(enhanced_img)
                # Process the image and set it to the label
                pixmap = self.process_image_to_show(image_instance.Qimg)
                label.setPixmap(pixmap)
            except Exception as e:
                logging.error(f"change_image_property: {e}")
                tb2 = traceback.format_exc()
                logging.error(tb2)

        def resize_images(self, new_img):
            try:
                # logging.debug(f"new_img_width = {new_img_width}, new_img_height = {new_img_height}")
                # Find the smallest image size
                for img in self.lbl_images_dic.values():
                    width, height = img.img_size
                    self.min_width = min(self.min_width, width)
                    self.min_height = min(self.min_height, height)
                new_size = (int(self.min_width), int(self.min_height))
                min_shape = new_size
                # Iterate over all images
                for label, img in self.lbl_images_dic.items():
                    # Update min_shape
                    for array in [img.magnitude, img.phase, img.real, img.imaginary]:
                        min_shape = tuple(min(m, s) for m, s in zip(min_shape, array.shape))
                    # Resize the image
                    img.Pimg = img.Pimg.resize((self.min_width, self.min_height))
                    img.original_img = img.original_img.resize((self.min_width, self.min_height))
                    img = img.update_image(img.Pimg)
                    logging.debug(f"images updated = {img.img_size}")
                # Now min_shape contains the shape of the smallest array
                # Resize all arrays to min_shape
                for label, img in self.lbl_images_dic.items():
                    for attr in ['Magnitude', 'phase', 'real', 'imaginary']:
                        array = getattr(img, attr)
                        zoom_factor = (new_size[0] / array.shape[0], new_size[1] / array.shape[1])
                        resized_array = zoom(array, zoom_factor)
                        setattr(img, attr, resized_array)
                new_img = new_img.update_image(new_img.Pimg)
                # logging.debug(f"new_img = {new_img.img_size}")
                # Return the new image resized
                return new_img
            except Exception as e:
                logging.error(f"Error2: {e}")
                tb2 = traceback.format_exc()
                logging.error(tb2)

        def display_ft_component(self, ft_label):
            try:
                # Get the corresponding image label
                img_label = self.img_ft_labels_dic[ft_label]
                # Get the QComboBox widget associated with the label
                combo_box = self.combo_boxes[ft_label]
                # Get the current text of the QComboBox widget
                text = combo_box.currentText()
                # Get the Image instance associated with the label
                image_instance = self.lbl_images_dic[img_label]
                # Based on the text, call the appropriate FT component from the Image instance
                selected_component = getattr(image_instance, text.lower())
                # Convert the numpy array to a QImage then, to a pixmap
                self.qft_img = qimage2ndarray.array2qimage(selected_component, normalize=True)
                pixmap = self.process_ft_image_to_show(ft_label)
                pixmap_item = QGraphicsPixmapItem()
                pixmap_item.setPixmap(pixmap)
                # Add the QGraphicsPixmapItem to the QGraphicsScene
                self.scenes[ft_label].clear()
                self.scenes[ft_label].addItem(pixmap_item)
                # Set the QGraphicsView's scene
                ft_label.setScene(self.scenes[ft_label])
                # # Set the pixmap to the label
                # ft_label.setPixmap(pixmap)
            except Exception as e:
                logging.error(f"Error: {e}")
                tb2 = traceback.format_exc()
                logging.error(tb2)

        def update_weights_lst(self, value, index):
            try:
                # Start a new thread for the components_mixer operation
                # self.worker = Worker(self.components_mixer, [], {})
                # self.worker.signals.progress.connect(self.update_progress_bar)
                # self.worker.start()
                # self.worker.signals.progress.emit(0)
                # # Emit progress signal
                # self.worker.signals.progress.emit(100)  # Emit signal with 100% progress

                label = self.img_indices_lbls_dic[index]
                img_obj = self.lbl_images_dic[label]
                combox = getattr(self, f"comboBox_img{index}")
                component_index = self.components_indices_dic[combox.currentText()]
                # Normalize the slider value to be between 0 and 1
                normalized_value = value / 100.0
                # logging.debug(f"slider value = {value}")
                # logging.debug(f"normalized_value = {normalized_value}")
                self.weights_lst[index - 1] = normalized_value
                # img_obj.weights_lst[component_index] = normalized_value
                self.components_mixer()
            except Exception as e:
                logging.error(f"update_weights_lst: {e}")
                tb2 = traceback.format_exc()
                logging.error(tb2)

        def update_progress_bar(self, value):
            self.progressBar.setValue(value)

        def components_mixer(self):
            # I need to know which objects that the user changed: which slider or combo box
            try:
                is_RadioButton1 = "Magnitude" in self.comboxItems
                for component in ['Magnitude', 'phase', 'real', 'imaginary']:
                    self.weighted_sums[component] = np.zeros((int(self.min_width), int(self.min_height)),
                                                             dtype='complex_')
                mood_components = ['magnitude', 'phase'] if is_RadioButton1 else ['real', 'imaginary']
                self.used_indices = []
                for index, label in self.img_indices_lbls_dic.items():
                    if label in self.lbl_images_dic.keys():
                        self.used_indices.append(index)
                        # [1, 3]
                print(f"used_indices: {self.used_indices}")

                for index in self.used_indices:  # we need combox component, label image, slider weight
                    print(f"index: {index}")
                    label = getattr(self, f"label_img{index}")
                    img = self.lbl_images_dic[label]
                    combox = getattr(self, f"comboBox_img{index}")
                    comp = combox.currentText().lower()
                    if comp == "magnitude":
                        comp = "Magnitude"
                    print(f"comp: {comp}")
                    if self.roi_rect is None:
                        img_comp = getattr(img, comp)
                    else:
                        logging.debug(f"images_roi keys {self.images_roi.keys()}")
                        img_comp = self.images_roi[index]
                        print(img_comp.shape)
                    weight = self.weights_lst[index - 1]
                    print(f"weight: {weight}")
                    self.weighted_sums[comp] += weight * img_comp
                    print(f"weighted_sums[{comp}] shape: {self.weighted_sums[comp].shape}")

                if is_RadioButton1:
                    # self.weighted_sums['magnitude'] = np.exp(self.weighted_sums['magnitude'] / 20)
                    fft = self.weighted_sums['Magnitude'] * np.exp(1j * self.weighted_sums['phase'])
                    cv2.imwrite('test.jpg', np.real(np.fft.ifft2(np.fft.ifftshift(fft))))
                else:
                    fft = self.weighted_sums['real'] + 1j * self.weighted_sums['imaginary']
                    cv2.imwrite('test.jpg', np.real(np.fft.ifft2(np.fft.ifftshift(fft))))
                inverse_fft = np.real(np.fft.ifft2(np.fft.ifftshift(fft)))
                # Emit progress signal here
                self.worker = Worker(lambda: None, [], {})  # Create a Worker instance as an attribute of MainApp
                progress = 100
                self.worker.signals.progress.emit(progress)  # Emit signal from the Worker instance
                if getattr(self, f"comboBox_outputs").currentText() == "Output 1":
                    self.label_output1.clear()
                    # pw = pg.PlotWidget()
                    # pw.getAxis('bottom').hide()
                    image = cv2.imread(r'F:\task 4\test.jpg')
                    self.label_output1.setImage(image.T)
                else:
                    self.label_output2.clear()
                    image = cv2.imread(r'F:\task 4\test.jpg')
                    self.label_output2.setImage(image.T)
            except Exception as e:
                logging.error(f"update_output_image: {e}")
                tb2 = traceback.format_exc()
                logging.error(tb2)

        def calc_roi(self, rect):
            if not self.is_full_region():
                # Convert rectangle coordinates from label coordinates to image coordinates
                converted_rect = self.convert_coordinates(rect)
                x = converted_rect.x()
                y = converted_rect.y()
                width = converted_rect.width()
                height = converted_rect.height()
                logging.debug(f"x: {x}")
                logging.debug(f"y: {y}")
                logging.debug(f"width: {width}")
                logging.debug(f"height: {height}")

                self.used_indices = []
                for index, label in self.img_indices_lbls_dic.items():
                    if label in self.lbl_images_dic.keys():
                        self.used_indices.append(index)
                # key: used indices , value: roi
                for index in self.used_indices:
                    logging.debug(f"calc_roi, index: {index}")
                    label = getattr(self, f"label_img{index}")
                    img = self.lbl_images_dic[label]
                    combox = getattr(self, f"comboBox_img{index}")
                    comp = combox.currentText().lower()
                    if comp == "magnitude":
                        comp = "Magnitude"
                    logging.debug(f"calc_roi, comp: {comp}")
                    img_comp = getattr(img, comp)
                    # img_comp[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()]
                    if self.comboBox.currentText() == 'Inner region':
                        mask = np.zeros_like(img_comp, dtype=bool)
                        logging.debug(f"mask: {mask.shape}")
                        mask[y:y + height, x:x + width] = True
                        logging.debug(f"mask roi: {mask[y:y + height, x:x + width].shape}")
                        # Take a slice excluding the specified region
                        self.images_roi[index] = img_comp * mask
                        # self.images_roi[index] = img_comp[y:y+height, x:x+width]
                        logging.debug(f"inner shape: {self.images_roi[index].shape}")
                    else:
                        mask = np.ones_like(img_comp, dtype=bool)
                        mask[y:y + height, x:x + width] = False
                        # Take a slice excluding the specified region
                        self.images_roi[index] = img_comp * mask
                        logging.debug(f"outer shape: {self.images_roi[index].shape}")
            self.components_mixer()

        def is_full_region(self):
            if self.comboBox.currentText() == 'Full region':
                self.roi_rect = None
                self.images_roi = {}
                for viewer in self.ft_labels:
                    if viewer.scene() is not None:
                        viewer.rubberBand.hide()
                return True
            else:
                return False

        def convert_coordinates(self, rect):
            labels_width = self.label_img1_fft.width()
            labels_height = self.label_img1_fft.height()
            images_width = self.min_width
            images_height = self.min_height

            scale_factor_x = images_width / labels_width
            scale_factor_y = images_height / labels_height

            x = round(rect.x() * scale_factor_x)
            y = round(rect.y() * scale_factor_y)
            width = round(rect.width() * scale_factor_x)
            height = round(rect.height() * scale_factor_y)

            return QRect(x, y, width, height)

        def process_ft_image_to_show(self, ft_label):
            # # Log the value of qimg
            # if qft_img is None:
            #     logging.debug("qft_img is None")
            # else:
            #     logging.debug(f"{ft_label} size: {qft_img.size()}")
            if self.qft_img:
                # Resize the QImage to fit the label
                self.qft_img = self.qft_img.scaled(ft_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logging.debug(f"qft_img: {self.qft_img.size()}")
                # Convert QImage to QPixmap
                pixmap = QPixmap.fromImage(self.qft_img)
                pixmap_item = QGraphicsPixmapItem()
                pixmap_item.setPixmap(pixmap)

                return pixmap
            else:
                logging.warning("FT Image Not Found!")
                return None

        def process_image_to_show(self, qimg):
            label, img = next(((k, v) for k, v in self.lbl_images_dic.items() if v.Qimg == qimg), (None, None))
            if img and qimg:
                # Resize the QImage to fit the label
                qimg = qimg.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # Convert QImage to QPixmap
                pixmap = QPixmap.fromImage(qimg)
                return pixmap
            else:
                logging.warning("Image Not Found!")
                return None

        # logging.debug(f"rect in main: {rect}")
        # logging.debug(f"rect x: {rect.x()}")
        # logging.debug(f"rect y: {rect.y()}")
        # logging.debug(f"rect width: {rect.width()}")
        # logging.debug(f"rect height: {rect.height()}")
        # logging.debug(f"rect top left: {rect.topLeft()}")
        # logging.debug(f"rect top right: {rect.topRight()}")
        # logging.debug(f"rect bottom left: {rect.bottomLeft()}")
        # logging.debug(f"rect bottom right: {rect.bottomRight()}")
        # ---------------------
        # img1 = self.lbl_images_dic[self.label_img1]
        # logging.debug(f"img1: {img1}")
        # img2 = self.lbl_images_dic[self.label_img3]
        # logging.debug(f"img2: {img2}")
        # self.weighted_sums['magnitude'] = 1 * img1.magnitude
        # self.weighted_sums['phase'] = 1 * img2.phase
        # ---------------------
        # def show_image(self, image_path):
        #     # Display the image
        #     img = QPixmap(image_path)
        #     self.label.setPixmap(img)
        #     self.resize(img.width(), img.height())
        # ---------------------
        # labels_width = self.label_img1_fft.width()
        # labels_height = self.label_img1_fft.height()
        # logging.debug(f"labels_width: {labels_width}")
        # logging.debug(f"labels_height: {labels_height}")
        # logging.debug(f"images_width: {self.qft_img.width()}")
        # logging.debug(f"images_height: {self.qft_img.height()}")
        # scale_factor_x = self.qft_img.width() / labels_width
        # scale_factor_y = self.qft_img.height() / labels_height
        # logging.debug(f"scale_factor_x: {scale_factor_x}")
        # logging.debug(f"scale_factor_y: {scale_factor_y}")
        #
        # x = round(rect.x() * scale_factor_x)
        # y = round(rect.y() * scale_factor_y)
        # width = round(rect.width() * scale_factor_x)
        # height = round(rect.height() * scale_factor_y)
        # logging.debug(f"x: {x}")
        # logging.debug(f"y: {y}")
        # logging.debug(f"width: {width}")
        # logging.debug(f"height: {height}")
        # return QRect(x, y, width, height)
        # ---------------------
        # def change_brightness(self, label, event):
        #     self.change_image_property(label, event, ImageEnhance.Brightness, event.y() / label.height() + 0.4)
        #
        # def change_contrast(self, label, event):
        #     self.change_image_property(label, event, ImageEnhance.Contrast, event.x() / label.width() + 0.3)
        #
        # def change_image_property(self, label, event, enhancer_class, value):
        #     logging.debug(f"event: {event}, enhancer_class: {enhancer_class}, value: {value}")
        #     image_instance = self.images_dic[label]
        #     if event.buttons() == Qt.LeftButton:
        #         enhancer = enhancer_class(image_instance.Pimg)
        #         enhanced_im = enhancer.enhance(value)
        #         image_instance.update_image(enhanced_im)
        #         # Process the image and set it to the label
        #         pixmap = self.process_image_to_show(image_instance.Qimg)
        #         label.setPixmap(pixmap)

        # def resize_images(self, new_qimg):
        #     try:
        #         # Get the size of the new image
        #         new_qimg_size = new_qimg.size()
        #         print(f"new_qimg_size = {new_qimg_size}")
        #         # Find the smallest image size
        #         self.smallest_size = min(img.Qimg.size().width() * img.Qimg.size().height()
        #                                  for img in self.images_dic.values())
        #         # Compare the new image size with the smallest size
        #         if (new_qimg_size.width() * new_qimg_size.height() <
        #                 self.smallest_size.width() * self.smallest_size.height()):
        #             self.smallest_size = new_qimg_size
        #             print(f"smallest_size = {self.smallest_size}")
        #         # Resize the new image
        #         new_qimg = new_qimg.scaled(self.smallest_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        #         # Resize all images to the smallest size
        #         for label in self.images_dic:
        #             img = self.images_dic[label]
        #             img.Qimg = img.Qimg.scaled(self.smallest_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        #         # Return the new image resized
        #         return new_qimg
        #     except Exception as e:
        #         print(f"Error2: {e}")
        #         tb2 = traceback.format_exc()
        #         print(tb2)


    if __name__ == "__main__":
        app = QtWidgets.QApplication(sys.argv)
        window = MainApp()
        window.show()
        sys.exit(app.exec_())
except Exception as e:
    print("Exception occurred:\n", traceback.format_exc())
