

function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('preview');
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
    const param = 'exemple';
    $.ajax({
        url: '/api/check_photo',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({param: param}),
        success: function(data) {
            // Traitez le résultat ici
            const res = document.getElementById('result')
            if (data.result) {
                res.innerHTML = 'True';
                res.style.backgroundColor = '#5dca38';
            }
            else {
                res.innerHTML = 'False';
                res.style.backgroundColor = '#ca3886';
            }
        },
        error: function(error) {console.error('Erreur Jquery:', error);}
    });
}