from flask import Flask, render_template, request
from tensorflow.keras import Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input,InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, add
import numpy as np
import cv2

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

max_length = 34
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))

se1 = Embedding(1652, 200, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(1652, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.load_weights("model_150.h5")

model1 = InceptionV3(weights='imagenet')
model_new = Model(model1.input, model1.layers[-2].output)

def load_doc(filename):
    # Opening file for read only
    file1 = open(filename, 'r')
    # read all text
    text = file1.read()
    # close the file
    file1.close()
    return text


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file = request.files['file1']
    file.save('static/file.jpg')

    image = preprocess('static/file.jpg')  # preprocess the image
    fea_vec = model_new.predict(image).reshape((1,2048))  # Get the encoding vector for the image


    data = load_doc('vocab.txt')
    vocab = data.split('/n')

    ixtoword = {}  # index to word
    wordtoix = {}  # word to index

    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict((fea_vec, sequence), verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    print(final)
    return render_template('predict.html', prediction=final)


if __name__ == "__main__":
    app.run(debug=True)



