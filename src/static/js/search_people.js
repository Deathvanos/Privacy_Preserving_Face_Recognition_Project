

document.addEventListener('DOMContentLoaded', function() {
    initializeToggleListener();
});

function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('upload-container');
            preview.src = e.target.result;
            preview.style.display = 'block';
            document.querySelector('.upload-label').style.display = 'none';
            // add buttons
            const buttons_preview = document.getElementById('photoActions');
            buttons_preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

function triggerFileInput() {
    document.getElementById('fileInput').click();
}

// TO DELETE
function checkPhoto() {
    // Get data
    const fileInput = document.getElementById('fileInput');
    let files;
    if (fileInput.files.length === 0 && typeof capturedFiles !== 'undefined') {
        files = capturedFiles
    }
    else {
        files = fileInput.files;
    }
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('fileInput', files[i]);
    }
    // Call the server
    const param = 'exemple';
    $.ajax({
        url: '/api/check_photo',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(data) {
            // Traitez le résultat ici
            const res = document.getElementById('result')
            console.log(data);
            res.innerHTML = `Prediction: ${data.prediction} | Trust: ${data.trust}`;
            if (data.trust >= 0.8) {
                res.style.backgroundColor = '#5dca38';
            }
            else {
                res.style.backgroundColor = '#ca3886';
            }
        },
        error: function(error) {console.error('Erreur Jquery:', error);}
    });
}


function initializeToggleListener() {
    const toggle = document.getElementById('toggle');
    let cameraScriptLoaded = false;

    toggle.addEventListener('change', function () {
        const upload_container = document.getElementById('upload-container');
        upload_container.src = "";
        upload_container.style.display  = "none";
        const buttons_preview = document.getElementById('photoActions');
        buttons_preview.style.display = 'none';

        if (toggle.checked) {
            // Toggle ON = Take Photos
            document.getElementById('import-photo-container').style.display = 'none';
            document.getElementById('take-photo-container').style.display = 'block';
            if (!cameraScriptLoaded) {
                const script = document.createElement('script');
                script.src = CAMERA_SCRIPT_URL
                script.onload = () => console.log('Camera script loaded.');
                document.body.appendChild(script);
                cameraScriptLoaded = true;
            }
        } else {
            // Toggle OFF = Import Photos
            document.getElementById('import-photo-container').style.display = 'block';
            document.getElementById('take-photo-container').style.display = 'none';
            const preview = document.getElementById('upload-label');
            preview.style = '';
        }
    });
}


// Lier le bouton Capture
captureBtn = document.getElementById('capture-btn');
captureBtn.addEventListener('click', () => {
    const maxPhotos = 1;  // ici tu peux changer dynamiquement
    const delay = 100;     // délai entre les captures en ms
    startPhotoCapture(maxPhotos, delay);
    // add buttons
    const buttons_preview = document.getElementById('photoActions');
    buttons_preview.style.display = 'block';

});