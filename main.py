import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacija za segmentaciju slike")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(expand=True, fill="both")

        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.pack(pady=10)

        self.left_panel = ctk.CTkCanvas(self.image_frame, bg="#333333", relief="solid", bd=1)
        self.left_panel.grid(row=0, column=0, padx=5, pady=5)

        self.right_panel = ctk.CTkCanvas(self.image_frame, bg="#333333", relief="solid", bd=1)
        self.right_panel.grid(row=0, column=1, padx=5, pady=5)

        self.frame_load_button = ctk.CTkFrame(self.main_frame)
        self.frame_load_button.pack(pady=5)

        self.btn_load = ctk.CTkButton(self.frame_load_button, text="Učitaj sliku", command=self.load_image, width=150, font=("verdana", 10, "bold"))
        self.btn_load.pack(padx=10, pady=10)

        self.frame_segmentation = ctk.CTkFrame(self.main_frame)
        self.frame_segmentation.pack(pady=20)

        self.segmentation_label = ctk.CTkLabel(self.frame_segmentation, text="Izaberi segmentaciju", font=("verdana", 10, "bold"))
        self.segmentation_label.grid(row=0, column=0, padx=10, pady=10)

        self.segmentation_method = ctk.CTkComboBox(self.frame_segmentation, values=["Thresholding", "GrabCut", "Canny Edge"], width=150)
        self.segmentation_method.grid(row=1, column=0, padx=10, pady=10)

        self.btn_apply = ctk.CTkButton(self.frame_segmentation, text="Primijeni segmentaciju", command=self.apply_segmentation, width=150, font=("verdana", 10, "bold"))
        self.btn_apply.grid(row=2, column=0, padx=10, pady=10)

        self.image_path = ""
        self.img = None  # Initialize image variable to None

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            # Load image using OpenCV
            self.img = cv2.imread(self.image_path)
            self.update_panels(self.img)  # Update the panels to fit the image

    def update_panels(self, img):
        if img is not None:
            img_height, img_width, _ = img.shape

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            max_panel_width = screen_width // 2
            max_panel_height = screen_height // 2

            scale_width = max_panel_width / img_width
            scale_height = max_panel_height / img_height
            scale = min(scale_width, scale_height)

            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

           
            self.left_panel.config(width=new_width, height=new_height)
            self.right_panel.config(width=new_width, height=new_height)

            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            photo_image = ImageTk.PhotoImage(image)
            
            self.update_image(photo_image, self.left_panel)

    def update_image(self, photo_image, panel):
        # Clear the canvas and display the new image
        panel.delete("all")
        panel.create_image(0, 0, anchor="nw", image=photo_image)
        panel.config(scrollregion=panel.bbox("all"))  # Update scroll region to encompass the new image
        panel.image = photo_image  # Keep a reference to avoid garbage collection

    def apply_segmentation(self):
        if self.img is None:
            return  # If no image is loaded, do nothing
        method = self.segmentation_method.get()
        if method == "Thresholding":
            self.threshold_segmentation()   	
        elif method == "GrabCut":
            self.grab_cut()
        elif method == "Canny Edge":
            self.canny_edge()

    def threshold_segmentation(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        thresh_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.update_image_from_segmentation(thresh_img, self.right_panel)

    def grab_cut(self):
        rect = cv2.selectROI("Select ROI", self.img, fromCenter=False, showCrosshair=True)
        mask = np.zeros(self.img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        grabcut_img = self.img * mask2[:, :, np.newaxis]
        cv2.destroyAllWindows()
        self.update_image_from_segmentation(grabcut_img, self.right_panel)

    def canny_edge(self):
        edges = cv2.Canny(self.img, 100, 200)
        edge_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.update_image_from_segmentation(edge_img, self.right_panel)

    def update_image_from_segmentation(self, image, panel):
        # Resize the image to fit the panel size
        img_height, img_width, _ = image.shape
        panel_width = panel.winfo_width()
        panel_height = panel.winfo_height()
        
        # Scale the image to fit the panel while preserving the aspect ratio
        scale_width = panel_width / img_width
        scale_height = panel_height / img_height
        scale = min(scale_width, scale_height)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        photo_image = ImageTk.PhotoImage(image)
        
        self.update_image(photo_image, panel)

# Initialize and run the application
if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageSegmentationApp(root)
    root.mainloop()
