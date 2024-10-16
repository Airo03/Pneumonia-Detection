import gradio as gr
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load your trained model
model = load_model('basic_cnn_best_weights.hdf5')  # Replace with your model

# Define the ImageDataGenerator for rescaling
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

def classify_image(inp):
    # Convert the input image to a numpy array and make a copy of it
    inp = np.array(inp).copy()
    
    # Resize the input image
    inp = np.resize(inp, (150, 150, 3))
    
    # Convert the data type of the numpy array to float32
    inp = inp.astype('float32')
    
    # Expand dimensions to match the input shape of the model
    inp = np.expand_dims(inp, axis=0)
    
    # Apply the rescaling
    inp = test_datagen.standardize(inp)

    # Make predictions using the model
    preds = model.predict(inp)
    
    # Get the predicted class
    pred_class = "PNEUMONIA" if preds[0][0] >= 0.5 else "NORMAL"  # Replace with your classes

    return pred_class

# CSS custom untuk mengatur warna latar belakang dan tombol
custom_css = """
body {
    background-color: white;
}
.gr-button-primary {
    background-color: #ADD8E6;
    color: white;
}
.gr-box {
    background-color: #ADD8E6;
    border: 3px solid #ADD8E6;
}
"""

# Define the Gradio interface
iface = gr.Interface(fn=classify_image, 
                     inputs=gr.inputs.Image(type="numpy", label="Input Image"), 
                     outputs=gr.outputs.Label(label="Prediction"), 
                     theme='huggingface')
iface.launch()

