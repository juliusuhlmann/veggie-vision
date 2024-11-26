import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import predict as pred
import os

# Set the working directory to the current file's directory
os.chdir(os.path.dirname(__file__))

# Initialize Tkinter window
root = tk.Tk()
root.title("Veggie Vision")
root.geometry("600x600")
root.configure(bg="#222222")  # Black background

model_path = '../models/veggie_vision_tensorflow.h5'  # Model path

# Set default font to Courier
root.option_add("*Font", "Courier 12")

# Function to process and display an image
def process_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        # Load and resize image for display
        image = Image.open(file_path)
        image.thumbnail((400, 400))  # Resize to fit display
        img_tk = ImageTk.PhotoImage(image)
        
        # Display the image in the label
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Keep reference

        # Predict and display the result
        try:
            prediction = pred.predict_image(file_path, model_path)
            result_label.config(text=f"Estimated weight: {prediction:.0f}g", fg="#4CAF50")  # Success color
        except Exception:
            result_label.config(text="Error: Unable to make prediction", fg="#D32F2F")  # Error color
            messagebox.showerror("Prediction Error", "There was an error processing the image. Please try again.")

# Header with reduced padding
header = tk.Label(root, text="Veggie Vision", font=("Courier", 24, "bold"), bg="#222222", fg="#FFFFFF")
header.pack(pady=(5, 0))  # Reduced padding to minimize top space

# Canvas for rounded button with adjusted dimensions and no padding
def create_rounded_button(canvas, x, y, w, h, r, text, command=None):
    # Draw rounded rectangle
    canvas.create_arc(x, y, x + 2*r, y + 2*r, start=90, extent=90, fill="#4CAF50", outline="#4CAF50")
    canvas.create_arc(x + w - 2*r, y, x + w, y + 2*r, start=0, extent=90, fill="#4CAF50", outline="#4CAF50")
    canvas.create_arc(x, y + h - 2*r, x + 2*r, y + h, start=180, extent=90, fill="#4CAF50", outline="#4CAF50")
    canvas.create_arc(x + w - 2*r, y + h - 2*r, x + w, y + h, start=270, extent=90, fill="#4CAF50", outline="#4CAF50")
    canvas.create_rectangle(x + r, y, x + w - r, y + h, fill="#4CAF50", outline="#4CAF50")
    canvas.create_rectangle(x, y + r, x + w, y + h - r, fill="#4CAF50", outline="#4CAF50")
    
    # Add button text
    button_text = canvas.create_text(x + w/2, y + h/2, text=text, fill="white", font=("Courier", 14, "bold"))
    
    # Bind command to all items
    if command:
        canvas.tag_bind("button", "<Button-1>", lambda e: command())

# Create a canvas with adjusted height and no padding
canvas_height = 70  # Height adjusted to fit the button snugly
canvas = tk.Canvas(root, bg="#222222", highlightthickness=0, height=canvas_height)
canvas.pack(pady=0)

# Position the button closer to the top-left corner
button_x = 100  # Centered horizontally
button_y = 5    # Moved up vertically
button_width = 200
button_height = 50
button_radius = 25

# Create the rounded button
create_rounded_button(canvas, x=button_x, y=button_y, w=button_width, h=button_height, r=button_radius, text="Upload Image", command=process_image)

# Tag all items as "button" for event binding
canvas.addtag_all("button")

# Image display area with reduced padding
image_label = tk.Label(root, bg="#333333", width=400, height=400)
image_label.pack(pady=(5, 10))

# Result label with reduced padding
result_label = tk.Label(root, text="Upload a vegetable image for a weight estimate.",
                        font=("Courier", 12), bg="#222222", fg="#AAAAAA")
result_label.pack(pady=(0, 10))

# Footer with reduced padding
footer = tk.Label(root, text="© 2024 VeggieVision | Julius Uhlmann", font=("Courier", 10),
                  bg="#222222", fg="#AAAAAA")
footer.pack(pady=(5, 5))

# Run the application
root.mainloop()
