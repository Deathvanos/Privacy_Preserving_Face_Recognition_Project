// ----------------------------------------------------------------------//
// ----------------------------// ON LOAD //-----------------------------//
// ----------------------------------------------------------------------//

document.addEventListener('DOMContentLoaded', function() {
    initializeStepClickListener();
    setCurrentStep(1);
    initializeFileUploadListener();
    initializeToggleListener();
    initializeImageSizeControl('img_size_value', 'img_size_unit', 100, 2500);
    initializeImageSizeControl('k_pixel', 'k_pixel_unit', 4, 1000);
    initializeImageSizeControl('pca_components', 'pca_components_unit', 11, 1000);
    initializeImageSizeControl('privacyBudget', 'privacyBudget_unit', 0.1, 1000);
});

function setCurrentStep(stepNumber) {
    const steps = document.querySelectorAll('.arrow-steps .step');
    const panels = document.querySelectorAll('.panel');
    steps.forEach((step, index) => {
        if (index === stepNumber - 1) {
            step.classList.add('current');
        } else {
            step.classList.remove('current');
        }
    });
    panels.forEach((panel, index) => {
        if (index === stepNumber - 1) {
            panel.classList.remove('hidden');
        } else {
            panel.classList.add('hidden');
        }
    });
}

function initializeStepClickListener() {
    const steps = document.querySelectorAll('.arrow-steps .step');
    steps.forEach((step, index) => {
        step.addEventListener('click', function() {
            setCurrentStep(index + 1);
        });
    });
}

function initializeFileUploadListener() {
    document.getElementById('fileInput').addEventListener('change', function (event) {
        const files = event.target.files;
        const container = document.getElementById('upload-container');
        container.innerHTML = ''; // Clear previous previews
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const reader = new FileReader();
            reader.onload = function (e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'upload-image';
                container.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    });
}


// ----------------------------------------------------------------------//
// ------------------------// PROCESS STEP //----------------------------//
// ----------------------------------------------------------------------//

function step_upload(go_next=false) {
    // Get data
    const fileInput = document.getElementById('fileInput');
    let files;
    if (fileInput.files.length === 0 && typeof capturedFiles !== 'undefined') {
        files = capturedFiles
    }
    else {
        files = fileInput.files;
    }
    const img_size_value = document.getElementById('img_size_value');
    const img_size_unit = document.getElementById('img_size_unit');
    const formData = new FormData();
    formData.append('img_size_value', img_size_value.value);
    formData.append('img_size_unit', img_size_unit.value);
    for (let i = 0; i < files.length; i++) {
        formData.append('fileInput', files[i]);
    }
    // Prepare success method
    function success_method(response, step) {
        display_image(response.images, step);
        if (go_next) {
            setCurrentStep(2)
            step_same_pixel()
        }
    }
    // Call the server
    call_process('1', success_method, formData);
}

function step_same_pixel(go_next=false) {
    // Get data
    const k_same_value = document.getElementById('k_pixel');
    const formData = new FormData();
    formData.append('k_same_value', k_same_value.value);
    // Prepare success method
    function success_method(response, step) {
        display_image(response.images, step-1);
        if (go_next) {

            setCurrentStep(3)
            step_pca()
        }
    }
    // Call the server
    call_process('2', success_method, formData);
}


function step_pca(go_next=false) {
    // Get data
    const param_pca_components = document.getElementById('pca_components');
    const formData = new FormData();
    formData.append('pca_components', param_pca_components.value);
    // Prepare success method
    function success_method(response, step) {
        display_image(response.images, step);
        ele = document.getElementById('image-container-' + step);
        ele.innerHTML += "<br><br>Eigenface Images are not linked with Eigenface Vectors. This is just a visual representation of the process. The next step will give you the initial number of images.";


        if (go_next) {
            setCurrentStep(4)
            step_noise()
        }
    }
    // Call the server
    call_process('3', success_method, formData);
}


function step_noise(go_next=false) {
    // Get data
    const param_epsilon = document.getElementById('privacyBudget');
    const formData = new FormData();
    formData.append('epsilon', param_epsilon.value);
    // Prepare success method
    function success_method(response, step) {
        display_image(response.images, step);
        if (go_next) {
            setCurrentStep(5)
        }
    }
    // Call the server
    call_process('4', success_method, formData);
}



function step_save(go_next=false) {
    // Get data
    //...
    // Prepare success method
    function success_method(response, step) {
        htmlContent = "New user created. His identification number is: " + response.user_id
        document.getElementById('user_id').innerHTML = htmlContent;
    }
    // Call the server
    call_process('5', success_method);
}

ML_RESULT = ""
function step_ML(go_next=false) {
    ml_timer = document.getElementById('ml_timer')
    timer = 0

    // Lance un timer qui met à jour le message chaque seconde
    const countdownInterval = setInterval(() => {
        timer += 1;
        ml_timer.innerHTML = `Please wait (approx. 200 seconds), the model is training... <br>${timer}`;
        // Facultatif : stoppe l'affichage si ça descend à 0
        if (timer > 240) {
            clearInterval(countdownInterval);
            ml_timer.innerHTML = `Training took longer than expected... waiting for results.`;
        }
    }, 1000);
    // Get data
    //...
    // Prepare success method
    function success_method(response, step) {
        ML_RESULT = response
        clearInterval(countdownInterval);
        displayBase64Images([response.curves], "ml_result_graph")
        displayBase64Images([response.classification_report], "ml_result_table")
        ml_result = document.getElementById('ml_result')
        ml_result.innerHTML = "accuracy: "+ Math.round(response.evaluation.accuracy*100)/100 + ", loss: " + Math.round(response.evaluation.loss*100)/100;

    }
    // Call the server
    call_process('6', success_method);
}




function call_process(step, success_method, formDataBase=null) {
    // Create & merge formData
    const formData = new FormData();
    formData.append('step', step);
    if (formDataBase) {
        for (const [key, value] of formDataBase.entries()) {
            formData.append(key, value);
        }
    }
    // Call the server
    const xhr = $.ajax({
        url: '/new_people',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            console.log(response);
            // Delete error if exist
            set_error();
            // Do the success process
            success_method(response, step)
        },
        error: function (error) {
            set_error(error.responseJSON ? error.responseJSON.error : 'Unknown error');
        }
    });
    // Annuler la requête si l'utilisateur quitte la page
    window.addEventListener("beforeunload", () => {
        xhr.abort();
    });
}

function set_error(text='') {
    document.getElementById('error-up').innerHTML = `${text}`;
    document.getElementById('error-down').innerHTML = `${text}`;
}
// ----------------------------------------------------------------------//
// ---------------------------// UTILS //--------------------------------//
// ----------------------------------------------------------------------//



function display_image(images, step) {
    if (images) {
        let htmlContent = '';
        images.forEach(image => {
            htmlContent += `<img src="data:image/jpeg;base64,${image}" alt="Image">`;
        });
        htmlContent += '';
        document.getElementById('image-container-' + step).innerHTML = htmlContent;
    }
}



function initializeToggleListener() {
    const toggle = document.getElementById('toggle');
    let cameraScriptLoaded = false;

    toggle.addEventListener('change', function () {
        if (toggle.checked) {
            // Toggle ON = Take Photos
            document.getElementById('import-photos-container').style.display = 'none';
            document.getElementById('take-photos-container').style.display = 'block';
            if (!cameraScriptLoaded) {
                const script = document.createElement('script');
                script.src = CAMERA_SCRIPT_URL
                script.onload = () => console.log('Camera script loaded.');
                document.body.appendChild(script);
                cameraScriptLoaded = true;
            }
        } else {
            // Toggle OFF = Import Photos
            document.getElementById('import-photos-container').style.display = 'block';
            document.getElementById('take-photos-container').style.display = 'none';
        }
    });
}


function initializeImageSizeControl(valueId, unitID, start, maxIntValue) {
    const inputValue = document.getElementById(valueId);
    const inputUnit = document.getElementById(unitID);

    const BASE_SIZE = start;   // ref
    const MAX_PX = maxIntValue;     // Max int
    const MAX_PERCENT = 100; // Max %

    inputValue.value = start;

    function updateLimits() {
        if (inputUnit.value === 'percent') {
            console.log(inputValue.value , MAX_PERCENT)
            if (inputValue.value > MAX_PERCENT) {
                inputValue.value = MAX_PERCENT;
            }
        } else {
            inputValue.max = MAX_PX;
        }
    }
    // Call when unit change
    inputUnit.addEventListener('change', function () {
        updateLimits();
    });
    // Init change reader
    updateLimits();
}


// Lier le bouton Capture
captureBtn = document.getElementById('capture-btn');
captureBtn.addEventListener('click', () => {
    const maxPhotos = 10;  // ici tu peux changer dynamiquement
    const delay = 100;     // délai entre les captures en ms
    startPhotoCapture(maxPhotos, delay);
});



// Function to render a list of base64 images
function displayBase64Images(base64List, id_balise) {
    const container = document.getElementById(id_balise);
    container.innerHTML = "";
    base64List.forEach((base64Str, index) => {
        const img = document.createElement("img");
        img.src = `data:image/jpeg;base64,${base64Str}`
        img.alt = `Image ${index + 1}`;
        container.appendChild(img);
    });
}