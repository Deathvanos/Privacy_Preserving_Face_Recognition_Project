# Privacy-preserving face recognition

Flask GUI in Python for anonymize a photo and retrieve it in a database.

## Description

This project will explore the intersection of Machine Learning (ML) and data privacy.
The student will investigate data anonymization techniques, such as differential privacy and k-anonymity, to enhance the privacy of ML models for facial recognition.
The aim of the project is the development a prototype that take a photo and match it with the one in the anonymized database.

## Setup

```shell
$ git clone https://github.com/Deathvanos/Privacy_preserving_face_recognition
$ cd Privacy_Preserving_Face_Recognition_Project/
$ python -m venv venv
$ .\venv\Scripts\activate
$ pip install -r requirements.txt
$ flask run
```

# Software architecture

The GUI files are organized into six folders:
- The folder ``templates``, which contains standard HTML pages (frontend)
- The folder ``static/js``, which contains JavaScript functions used by a single page
- The folder ``static/css``, which contains style pages used by a single page
- The folder ``static/templates``, which contains pages common to multiple pages (ex: header/footer page).
- The ``static/assets`` folder, which contains external resources usable by the program (images, fonts, PDF documents, etc.)
- The folder ``modules``, which contains all Python programs used by the server (backend)