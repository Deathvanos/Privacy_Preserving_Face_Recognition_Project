// Access the DOM elements
const video = document.getElementById('camera');
const canvas = document.getElementById('photo');
let captureBtn = document.getElementById('capture-btn');
const context = canvas.getContext('2d');

// Create a container to show the captured images
const gallery = document.createElement('div');
gallery.id = 'photo-gallery';
captureBtn.parentNode.insertBefore(gallery, captureBtn.nextSibling);  // Insert just after the button
let capturedFiles = [];

// Request access to the user's camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((error) => {
        console.error("Error accessing the camera: ", error);
        const toggle = document.getElementById('toggle');
        if (toggle) {
            toggle.checked = false;  // Revenir à "Import Photos"
            document.getElementById('import-photos-container').style.display = 'block';
            document.getElementById('take-photos-container').style.display = 'none';
        }
        alert("Unable to access the camera. Please check your permissions or device settings.");

    });


// Function to capture a photo
function capturePhoto() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    // resize
    const oval = document.getElementById('oval-overlay');
    const rect = oval.getBoundingClientRect();
    const videoRect = video.getBoundingClientRect();
    const cropX = rect.left - videoRect.left + rect.width/5;
    const cropY = rect.top - videoRect.top + rect.height/5;
    const cropWidth = rect.width+(rect.width/3);
    const cropHeight = rect.height+(rect.height/4);

    // Canvas Cropper
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = cropWidth;
    croppedCanvas.height = cropHeight;
    const croppedContext = croppedCanvas.getContext('2d');
    // Draw photo
    croppedContext.drawImage(
        canvas,
        cropX, cropY, cropWidth, cropHeight, // input
        0, 0, cropWidth, cropHeight // output
    );
    return croppedCanvas.toDataURL('image/jpeg');
}


// Fonction qui capture un certain nombre de photos
function startPhotoCapture(maxPhotos = 10, delay = 100) {
    clearImages(); // Clear previous photos
    let photosTaken = 0;

    const intervalId = setInterval(() => {
        if (photosTaken >= maxPhotos) {
            clearInterval(intervalId);
            console.log("Liste des fichiers capturés :", capturedFiles);
            return;
        }
        const imageData = capturePhoto();
        // Create base64 File
        const file = dataURLtoFile(imageData, `photo_${photosTaken}.jpg`);
        capturedFiles.push(file);
        // Print image
        const img = document.createElement('img');
        img.src = imageData;
        img.classList.add('upload-image');
        //img.style.width = '100px';   // You can adjust the size
        //img.style.margin = '5px';
        gallery.appendChild(img);
        gallery.classList.add('upload-container', "margin-top");
        // Counter
        photosTaken += 1;
    }, delay);
}

// Lier le bouton Capture
/*
captureBtn = document.getElementById('capture-btn');
captureBtn.addEventListener('click', () => {
    const maxPhotos = 10;  // ici tu peux changer dynamiquement
    const delay = 100;     // délai entre les captures en ms
    startPhotoCapture(MAX_PHOTOS, delay);
});
*/








function clearImages() {
    // Clear all previous photos
    capturedFiles = [];
    gallery.innerHTML = "";
    document.getElementById('upload-container').innerHTML = "";
    document.getElementById('fileInput').value = "";
}


// Convert base64 -> Blob -> File
function dataURLtoFile(dataurl, filename) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
}

