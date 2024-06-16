from flask import Flask, request, jsonify
import cv2
import base64
import pickle
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MongoDB Atlas connection string
mongo_uri = 'mongodb+srv://aksh9881:aksh9881@dev.zqib6ic.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true&appName=Dev'

client = MongoClient(mongo_uri)
db = client.get_database('fraud_detection')  # Replace with your database name
collection = db.get_collection('transactions')  # Replace with your collection name

# Function to encode faces
def encode_faces(faces):
    encoded_faces = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (100, 100))
        encoded_faces.append(resized_face.flatten())
    return encoded_faces

# Function to compare faces using cosine similarity
def compare_faces(face1, face2):
    return cosine_similarity([face1], [face2])[0][0]

# Route for storing face
@app.route('/store_face', methods=['POST'])
def store_face():
    try:
        if request.content_type != 'application/json':
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.json
        
        if 'image' not in data or 'user_id' not in data:
            return jsonify({'error': 'Missing image data or user_id'}), 400
        
        image_data = data['image']
        User_Id = data['user_id']
        
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        data_to_insert = {
            'user_id': User_Id,
            'image_base64': img_base64,
            'faces_detected': len(faces)
        }
        
        inserted_result = collection.insert_one(data_to_insert)
        
        return jsonify({
            'image': img_base64,
            'faces_detected': len(faces),
            'mongo_inserted_id': str(inserted_result.inserted_id)
        }), 200
    
    except Exception as e:
        app.logger.error(f"Error storing face: {e}")
        return jsonify({'error': 'An error occurred storing the face'}), 500


@app.route('/process_transaction', methods=['POST'])
def process_transaction():
    try:
        data = request.json
        
        # Extract necessary data from request
        image_base64 = data.get('image', '')
        User_Id = data.get('user_id', '')
        amount = data.get('amount', '')
        
        # Validate presence of required data
        if not image_base64 or not User_Id or not amount:
            return jsonify({'error': 'Missing required data: image, user_id, amount'}), 400
        
        # Decode base64 image and convert to PIL image
        image_bytes = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Example MongoDB query to retrieve stored document
        stored_document = collection.find_one({'user_id': User_Id})
        
        if stored_document:
            stored_image_base64 = stored_document.get('image_base64', '')
            stored_image_bytes = base64.b64decode(stored_image_base64)
            stored_image_pil = Image.open(io.BytesIO(stored_image_bytes))
            
            # Example comparison logic (adjust as needed)
            similarity_score = compare_images(image_pil, stored_image_pil)
            
            if similarity_score > 0.8:  # Example threshold
                return jsonify({'message': f'Transaction successful for amount {amount}.'}), 200
            else:
                return jsonify({'message': 'Transaction declined. Face not recognized.'}), 401
        else:
            return jsonify({'message': 'Stored document not found for user ID.'}), 404
    
    except Exception as e:
        app.logger.error(f"Error processing transaction: {e}")
        return jsonify({'error': 'An error occurred processing the transaction.'}), 500


@app.route('/process_payment', methods=['POST'])
def process_payment():
    try:
        if not request.content_type.startswith('multipart/form-data'):
            return jsonify({'error': 'Content-Type must be multipart/form-data'}), 400

        if 'image' not in request.files or 'amount' not in request.form:
            return jsonify({'error': 'Missing image or amount data'}), 400

        image_data = request.files['image'].read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            app.logger.warning("No face detected")
            return jsonify({'error': 'No face detected'}), 400

        encoded_faces = encode_faces(gray, faces)

        stored_faces = collection.find()

        for record in stored_faces:
            stored_face_data = base64.b64decode(record['image_base64'])
            stored_face_array = np.frombuffer(stored_face_data, np.uint8)
            stored_face_img = cv2.imdecode(stored_face_array, cv2.IMREAD_GRAYSCALE)
            
            if stored_face_img is None:
                continue
            
            stored_faces_detected = face_cascade.detectMultiScale(stored_face_img, scaleFactor=1.3, minNeighbors=5)
            
            if len(stored_faces_detected) == 0:
                continue

            stored_encoded_faces = encode_faces(stored_face_img, stored_faces_detected)

            for encoded_face in encoded_faces:
                for stored_face_encoded in stored_encoded_faces:
                    similarity = compare_faces(encoded_face, stored_face_encoded)
                    if similarity > 0.8:  # Assuming a threshold of 0.8 for face match
                        amount = request.form['amount']
                        return jsonify({'message': f'Payment of {amount} processed successfully', 'similarity': similarity}), 200

        return jsonify({'error': 'Face mismatch. Payment cannot be processed'}), 400

    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500
    
    
if __name__ == '__main__':
    app.run(debug=True)
