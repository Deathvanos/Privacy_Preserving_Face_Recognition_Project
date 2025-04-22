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
The student will investigate data anonymization techniques, such as differential privacy and k-anonymity, to enhance the privacy of ML models for facial recognition.
The aim of the project is the development a prototype that take a photo and match it with the one in the anonymized database.
"""
# ---------------------------------------------------------------------------
# Usefully links:
# * https://www.geeksforgeeks.org/single-page-portfolio-using-flask/
# * https://realpython.com/flask-blueprint/
# * https://www.geeksforgeeks.org/flask-rendering-templates/
# Usefully commands (pip install pipreqs)
# $ pipreqs  > requirements.txt
# $ poetry init # for generate .toml file
# ---------------------------------------------------------------------------
from flask import Flask, render_template, jsonify, request
from flask_assets import Environment, Bundle
import src.config as config
from controller.user_creation_controller import UserCreationController
from os import listdir

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
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

@app.route("/analysis")
def analysis_page():
    user_list = UserCreationController.get_user_list()
    return render_template("analysis.html", user_list=user_list)

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
            response, code = {'error': "ML not implemented"}, 400
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


@app.route("/recontruct_user", methods=['POST'])
def recontruct_user_action():
    print(request.form)
    print(request.files)
    print("recontruct_user called")
    user_id = request.form.get('user_id')
    # Return good execution message
    return jsonify({'result': 'end', 'user_id':user_id}), 200


@app.route('/api/check_photo', methods=['POST'])
def check_photo():
    import random
    return jsonify({'result': random.choice([True, False])})



# ---------------------------------------------------------------------------
# ------------------------- MAIN --------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

