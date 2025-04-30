
function load_yaleface() {
    function success_method(response, step) {
        document.getElementById('msg').innerHTML = "Yaleface dataset loaded";
    }
    // Call the server
    call_process("/api/load_yaleface", success_method);
}


function delete_db() {
    // Prepare success method
    function success_method(response, step) {
        document.getElementById('msg').innerHTML = "dataset deleted";
    }
    // Call the server
    call_process("/api/delete_db", success_method);
}

function load_lfw() {
    // Prepare success method
    function success_method(response, step) {
        document.getElementById('msg').innerHTML = "LFW dataset loaded";
    }
    // Call the server
    call_process("/api/load_lfw", success_method);
}

function call_process(url, success_method) {
    document.getElementById('msg').innerHTML = "Loading...";
    // Call the server
    const xhr = $.ajax({
        url: url,
        type: 'POST',
        success: function (response) {
            console.log(response);
            // Do the success process
            success_method(response)
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

