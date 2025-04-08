function load_user_data() {
    // Get data
    const param_selectBox = document.getElementById('selectBox');
    const formData = new FormData();
    formData.append('user_id', param_selectBox.value);
    // Prepare success method
    function success_method(response) {
        htmlContent = "<br>Identification number: " + response.user_id + "<br>Values:<br><br>";
        balise = document.getElementById('db_user')
        balise.innerHTML = htmlContent;
        // print bdd values
        displayBase64Images(response.user_data, "db_user_table")
        document.getElementById('user_container').hidden = false;
        set_error()
    }
    // Call the server
    call_process('/get_user_list', success_method, formData);
}


function delete_user_request() {
    // Get data
    const param_selectBox = document.getElementById('selectBox');
    const formData = new FormData();
    formData.append('user_id', param_selectBox.value);
    // Prepare success method
    function success_method(response) {
        set_error("User "+ response.user_id + " has been deleted. <br>The page will be reloaded in 3 seconds.");
        setTimeout(() => {window.location.reload();}, 3000);
    }
    // Call the server
    call_process('/delete_user', success_method, formData);
}





function call_process(url, success_method, formData=null) {
    // Call the server
    $.ajax({
        url: url,
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            console.log(response);
            // Delete error if exist
            set_error();
            // Do the success process
            success_method(response)
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


// Function to render the NumPy array
function displayNumpyArray(array, id_balise) {
    const table = document.createElement("table");
    table.className = 'table';
    array.forEach((row, rowIndex) => {
        const tr = document.createElement("tr");
        tr.className = 'cell';
        // Add an index
        const indexTd = document.createElement("td");
        indexTd.textContent = `Vector ${rowIndex + 1}`;
        indexTd.className = 'cell';
        tr.appendChild(indexTd);
        // Add values
        row.forEach(cell => {
            const td = document.createElement("td");
            td.textContent = cell;
            td.className = 'cell';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    // Append the table to a container (or body)
    const container = document.getElementById(id_balise);
    container.className = 'table-container';
    container.innerHTML = ""; // Clear any previous content
    container.appendChild(table);
}

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

function displayRowData(data, id_balise) {
    const container = document.getElementById(id_balise);
    container.innerHTML = "";
    data.forEach((item, index) => {
        const div = document.createElement("div");
        div.textContent = item;
        div.style.marginBottom = "8px";
        container.appendChild(div);
    });
}
