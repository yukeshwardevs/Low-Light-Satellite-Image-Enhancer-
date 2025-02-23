import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

def retinex(image, sigma_list):
    """
    Apply Retinex algorithm to enhance image.
    
    :param image: Input image (BGR format)
    :param sigma_list: List of sigma values for Gaussian blur
    :return: Retinex enhanced image
    """
    # Convert image to float32
    image = np.float32(image) + 1.0

    # Initialize the Retinex result
    retinex_result = np.zeros_like(image)

    for sigma in sigma_list:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        # Compute the Retinex result
        retinex_result += np.log(image) - np.log(blurred)

    # Normalize and convert back to uint8
    retinex_result = retinex_result / len(sigma_list)
    retinex_result = np.exp(retinex_result)
    retinex_result = cv2.normalize(retinex_result, None, 0, 255, cv2.NORM_MINMAX)
    retinex_result = np.uint8(retinex_result)

    return retinex_result

def enhance_feeble_light_signals(image, alpha, beta, clip_limit, gamma, sigma_list):
    # Apply Retinex enhancement
    retinex_image = retinex(image, sigma_list)

    # Convert to LAB color space
    lab_image = cv2.cvtColor(retinex_image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into channels
    l, a, b = cv2.split(lab_image)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel back with a and b channels
    lab_image_clahe = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)
    
    # Brighten the image by adjusting contrast (alpha) and brightness (beta)
    brightened_image = cv2.convertScaleAbs(enhanced_image, alpha=alpha, beta=beta)
    
    # Apply Gamma Correction
    gamma_corrected = np.power(brightened_image / 255.0, gamma)
    gamma_corrected = np.uint8(gamma_corrected * 255)
    
    return gamma_corrected

def process_image(input_image, alpha, beta, clip_limit, gamma):
    # Convert image to the format compatible with OpenCV
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Define sigma values for Retinex algorithm
    sigma_list = [15, 80, 250]  # You can adjust this as needed
    
    # Enhance the image using Retinex and other adjustments
    output_image = enhance_feeble_light_signals(input_image, alpha, beta, clip_limit, gamma, sigma_list)
    
    # Convert output image back to RGB for displaying
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    return output_image

# Define the Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Slider(minimum=1.0, maximum=10.0, value=3.0, label="Alpha (Contrast)"),
        gr.Slider(minimum=0, maximum=100, value=20, label="Beta (Brightness)"),
        gr.Slider(minimum=1.0, maximum=15.0, value=10.0, label="CLAHE Clip Limit"),
        gr.Slider(minimum=0.1, maximum=10.0, value=1.5, label="Gamma Correction"),
    ],
    outputs=gr.Image(type="numpy", label="Enhanced Image"),  # Only the enhanced image is shown
    title="Feeble Light Signal Image Enhancer",
    description="Upload a dark image, and enhance it using Retinex, CLAHE, contrast, brightness, and gamma correction."
)

# Launch the Gradio app
interface.launch()
