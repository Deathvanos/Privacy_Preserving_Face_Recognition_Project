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
# Usefully commands
# $ pip freeze > requirements.txt; poetry init
# ---------------------------------------------------------------------------
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory, session
from flask_assets import Environment, Bundle
import config
from modules.gui_controller import GUIController
from os import listdir


app = Flask(__name__)
app.config.from_object(config.Config)
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
    user_list = GUIController.get_user_list()
    return render_template("search_people.html", user_list=user_list)

@app.route("/show_database")
def show_database_page():
    user_list = GUIController.get_user_list()
    return render_template("show_database.html", user_list=user_list)

@app.route("/analysis")
def analysis_page():
    user_list = GUIController.get_user_list()
    return render_template("analysis.html", user_list=user_list)


@app.route("/new_people")
def new_people_init_page():
    #GUIController.delete_temp_file()
    return render_template("new_people.html")

@app.route("/new_people", methods=['POST'])
def new_people_processing_page():
    print(request.form)
    print(request.files)
    # Check step value
    step = request.form.get('step')
    if not step:
        return jsonify({'error': 'Step parameter is missing'}), 400
    try: step = int(step)
    except: return jsonify({'step': step, 'error': 'step is not an integer'}), 400
    # Initialisation of the Controller
    if step == 0:
        # Get user images
        files = request.files.getlist('fileInput')
        if not files:
            return jsonify({'error': 'No file part in the request'}), 400
        # Create new Controller:
        c = GUIController(files)
        c.save_to_file()
        return jsonify({'step':step, 'result': 'OK'})
    # Retrieve controller
    ctrl = GUIController.load_from_file()
    if not ctrl:
        return jsonify({'step':step, 'error': 'No controller initialized'}), 400
    # Do the requested action
    imgs = []
    if ctrl.can_run_step(int(step)):
        match step:
            case 1:
                ctrl.s1_apply_k_same_pixel()
                imgs = ctrl.get_image_pixelated("bytes")
            case 2:
                # Check width value
                width = request.form.get('width')
                if not width: return jsonify({'error': 'width parameter is missing'}), 400
                try: width = int(width)
                except: return jsonify({'step': step, 'error': 'width is not a int'}), 400
                # Check height value
                height = request.form.get('height')
                if not height: return jsonify({'error': 'height parameter is missing'}), 400
                try: height = int(height)
                except: return jsonify({'step': step, 'error': 'height is not a int'}), 400
                # Apply the process
                ctrl.s2_resize_images((width, height))
                imgs = ctrl.get_image_resized("bytes")
            case 3:
                # Check pca_components value
                pca_components = request.form.get('pca_components')
                if not pca_components: return jsonify({'error': 'pca_components parameter is missing'}), 400
                try: pca_components = int(pca_components)
                except: return jsonify({'step': step, 'error': 'pca_components is not a int'}), 400
                max_pca = ctrl.get_image_number()
                if pca_components > max_pca:
                    return jsonify({'step': step, 'error': f'pca_components should be between 0 and {max_pca}'}), 400
                # Apply the process
                ctrl.s3_generate_pca_components(pca_components)
                imgs = ctrl.get_image_eigenface("bytes")
            case 4:
                # Check epsilon value
                epsilon = request.form.get('epsilon')
                if not epsilon: return jsonify({'error': 'epsilon parameter is missing'}), 400
                try: epsilon = float(epsilon)
                except: return jsonify({'step': step, 'error': 'epsilon is not a float'}), 400
                # Apply the process
                i = ctrl.s4_apply_differential_privacy(epsilon)
                imgs = i + ctrl.get_image_noised("bytes")
            case 5:
                # Apply the process
                ctrl.s5_launch_ml()
            case 6:
                # Apply the process
                user_id = ctrl.s6_save_user()
                # No modification in the Controller, we can skeep now
                return jsonify({'step': step, 'result': 'end', 'user_id': user_id}), 200

    else:
         return jsonify({'step': step, 'error': "Can't run this step"}), 400


    # Save new modifications of the Controller
    ctrl.save_to_file()
    # Return good execution message
    return jsonify({'step':step, 'result': 'end', 'images':imgs}), 200


# ---------------------------------------------------------------------------
# ------------------------- BACK FUNCTIONS ----------------------------------
# ---------------------------------------------------------------------------

@app.route("/get_user_list", methods=['POST'])
def get_user_list_action():
    print(request.form)
    print(request.files)
    user_id = request.form.get('user_id')
    user_data = GUIController.get_user_data(int(user_id))
    # Return good execution message
    return jsonify({'result': 'end', 'user_id':user_id, "user_data":user_data.tolist()}), 200

@app.route("/delete_user", methods=['POST'])
def delete_user_action():
    print(request.form)
    print(request.files)
    print("delete_user called")
    user_id = request.form.get('user_id')
    result = GUIController.delete_user(int(user_id))
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

