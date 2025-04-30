# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Project     : Privacy-preserving face recognition
# Created By  : Elodie CHEN - Bruce L'HORSET - Charles MAILLEY
# Created Date: 11/02/2025
# Referent: Sara Ricci - ricci@vut.cz
# version ='1.0'
# ---------------------------------------------------------------------------
"""
This project will explore the intersection of Machine Learning (ML) and data privacy.
The student will investigate data anonymization techniques, such as differential privacy and k-anonymity, to enhance the privacy of ML ml_models for facial recognition.
The aim of the project is the development a prototype that take a photo and match it with the one in the anonymized database.
"""
# ---------------------------------------------------------------------------
# Usefully links:
# * https://www.geeksforgeeks.org/single-page-portfolio-using-flask/
# * https://realpython.com/flask-blueprint/
# * https://www.geeksforgeeks.org/flask-rendering-templates/
# Usefully commands (pip install pipreqs)
# $ pipreqs  > requirements.txt --force
# $ poetry init # for generate .toml file
# ---------------------------------------------------------------------------
from flask import Flask, render_template, jsonify, request
from flask_assets import Environment, Bundle
from os import listdir

from src.controller.ml_controller import MLController
from src.controller.user_creation_controller import UserCreationController
from src.controller.database_controller import DatabaseController
from src.modules.utils_image import pillow_image_to_bytes

app = Flask(__name__)
app.secret_key = SECRET_KEY = b'\x1f\x0e\x0c\xa6\xdbt\x01S\xa0$r\xf8$\xb4\xe3\x8a\xcf\xe0\\\x00M0H\x01'
# Configure SCSS bundle
assets = Environment(app)
assets.url = app.static_url_path
for filename in listdir(f"src/{assets.url}/css"):
    if filename.endswith('.scss'):
        name = filename[:-5]
        scss = Bundle(f"css/{filename}", filters='libsass', output=f'css/{name}.css')
        assets.register(f"scss_{name}", scss)



# ---------------------------------------------------------------------------
# ------------------------- WEB PAGE ----------------------------------------
# ---------------------------------------------------------------------------
@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/search_people")
def search_people_page():
    return render_template("search_people.html")

@app.route("/show_database")
def show_database_page():
    user_list = UserCreationController.get_user_list()
    return render_template("show_database.html", user_list=user_list)

@app.route("/dataset_loader")
def dataset_loader_page():
    return render_template("dataset_loader.html")

@app.route("/new_people")
def new_people_init_page():
    #GUIController.delete_temp_file()
    return render_template("new_people.html")


# ---------------------------------------------------------------------------
# ------------------------- BACK POST FUNCTIONS ----------------------------------
# ---------------------------------------------------------------------------


@app.route("/new_people", methods=['POST'])
def new_people_processing_page():
    # Print income values
    print(request.form)
    print(request.files)
    # Check step value
    try: step = int(request.form['step'])
    except KeyError: return jsonify({'error': 'Step parameter is missing'}), 400
    except (TypeError, ValueError): return jsonify({'error': 'Step parameter must be an integer'}), 400
    # Resolve the step number
    response, code = {'error': "step didn't match"}, 400
    match step:
        case 1:
            inputs = request.files.getlist('fileInput')
            inputs = inputs if inputs else None
            img_size_value = request.form.get('img_size_value')
            img_size_value = (img_size_value, img_size_value) if img_size_value else None
            img_size_unit = request.form.get('img_size_unit')
            img_size_unit = img_size_unit if img_size_unit else None
            response, code = UserCreationController.initialize_new_user(inputs, img_size_value, img_size_unit)
        case 2:
            value = request.form.get('k_same_value')
            value = value if value else None
            response, code = UserCreationController.apply_k_same_pixel(value)
        case 3:
            inputs = request.form.get('pca_components')
            response, code = UserCreationController.generate_pca_components(inputs)
        case 4:
            inputs = request.form.get('epsilon')
            response, code = UserCreationController.apply_differential_privacy(inputs)
        case 5:
            response, code = UserCreationController.save_user_in_db()
        case 6:
            mlc = MLController()
            try: mlc.prepare_data()
            except: return {'error': "Error in prepare_data"}, 400
            try: mlc.create_model()
            except: return {'error': "Error in create_model"}, 400
            try: result = mlc.train_model()
            except: return {'error': "Error in train_model"}, 400
            result['curves'] = pillow_image_to_bytes(result['curves'])
            result['confusion_matrix'] = result['confusion_matrix'].tolist()
            response, code = ({'duration': mlc.duration} | result, 200)
    return jsonify(response), code



@app.route("/get_user_list", methods=['POST'])
def get_user_list_action():
    print(request.form)
    print(request.files)
    user_id = request.form.get('user_id')
    user_data = UserCreationController.get_user_data(int(user_id))
    # Return good execution message
    return jsonify({'result': 'end', 'user_id':user_id, "user_data":user_data.tolist()}), 200

@app.route("/delete_user", methods=['POST'])
def delete_user_action():
    print(request.form)
    print(request.files)
    print("delete_user called")
    user_id = request.form.get('user_id')
    result = UserCreationController.delete_user(int(user_id))
    # Return good execution message
    return jsonify({'result': 'end', 'user_id':user_id, "nb_rows_delete": result}), 200



@app.route('/api/check_photo', methods=['POST'])
def check_photo():
    print(request.form)
    print(request.files)
    # Retrieve the image
    input = request.files.getlist('fileInput')
    if not input: return jsonify({'error': 'Please reload the page'}), 400
    input= input[0]
    image = MLController.convert_file_storage_to_numpy(input)
    print(image.shape)

    # Use the ML prediction
    mlc = MLController()
    result = mlc.predict_image(image)
    prediction, trust = int(result[0]), round(float(result[1]), 2)
    print(f"Prediction: {prediction}, Trust: {trust}")
    return jsonify({"prediction":prediction, "trust": trust}), 200


@app.route('/api/load_yaleface', methods=['POST'])
def load_yaleface():
    DatabaseController().load_yalefaces_dataset()
    return jsonify({"result": "ok"}), 200

@app.route('/api/load_lfw', methods=['POST'])
def load_lfw():
    DatabaseController().load_lfw_dataset()
    return jsonify({"result": "ok"}), 200

@app.route('/api/delete_db', methods=['POST'])
def delete_db():
    DatabaseController().reset_database()
    return jsonify({"result": "ok"}), 200





# ---------------------------------------------------------------------------
# ------------------------- MAIN --------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

