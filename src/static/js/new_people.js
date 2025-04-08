
// ----------------------------------------------------------------------//
// ----------------------------// ON LOAD //-----------------------------//
// ----------------------------------------------------------------------//

document.addEventListener('DOMContentLoaded', function() {
    initializeStepClickListener();
    setCurrentStep(1);
    initializeFileUploadListener();
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
    const img_size = document.getElementById('img_size');
    const files = fileInput.files;
    const formData = new FormData();
    formData.append('img_size', img_size.value);
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
            setCurrentStep(5)
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
            setCurrentStep(6)
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


function step_ML(go_next=false) {
    // Get data
    //...
    // Prepare success method
    function success_method(response, step) {}
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
    $.ajax({
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
}

function set_error(text='') {
    const errorContainer = document.getElementById('error');
    errorContainer.innerHTML = `${text}`;
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