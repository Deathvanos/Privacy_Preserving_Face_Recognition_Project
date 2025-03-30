from src.modules.peep import Peep
from pandas import DataFrame
from os import listdir
import numpy as np
from src.modules.utils_image import image_numpy_to_pillow
from functools import reduce
import operator
from PIL import Image, ImageDraw


def combine_pill_images_lists(image_lists, delimiter_color=(0, 0, 0), delimiter_width=10):
    """Combine a list of lists of PIL.Image.Images into a single image."""
    widths, heights = [], []
    for images in image_lists:
        w, h = zip(*(img.size for img in images))
        widths.append(sum(w) + (len(images) - 1) * delimiter_width)
        heights.append(max(h))
    total_width = max(widths)
    total_height = sum(heights) + (len(image_lists) - 1) * delimiter_width
    combined_image = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(combined_image)
    y_offset = 0
    for images in image_lists:
        x_offset = 0
        max_height = 0
        for img in images:
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img.width + delimiter_width
            max_height = max(max_height, img.height)
        y_offset += max_height + delimiter_width
    return combined_image


def import_subject_images(id):
    """Import images from a subject folder."""
    # Import images
    image_folder = "../../data/database"
    sujet = f"subject_{id}_"
    image_file = []
    for filename in listdir(image_folder):
        if filename.startswith(sujet):
            pillow_image = Image.open(f"{image_folder}/{filename}")
            image_file.append(pillow_image)
    return DataFrame({'userFaces':image_file, "userId": 16, "imageId":range(1, len(image_file)+1)})



def generate_noise_on_eigenface_images(subject=1, epsilon=5,
                   show_img:(bool,int) = False,
                   export_img = False, export_img_folder = "../../sandbox/noised_image_test",
                   ):
    """Workflow to generate noised images."""
    # Import images
    image_df = import_subject_images(subject)
    #print(len(image_df), image_df)
    if show_img: image_df['userFaces'][show_img].show() # show image source

    # Generate Eigenfaces [images are resized in this process]
    peep = Peep()
    peep.run_from_dataframe(image_df, epsilon)
    eigenfaces_list = peep.get_eigenfaces()
    #print(len(eigenfaces_list))
    #print(eg1.shape)
    #print(eg1.min(), eg1.max())
    if show_img:
        eg1 = eigenfaces_list[show_img]
        image_numpy_to_pillow(eg1).show()

    # Flatten Images
    img_shape = eigenfaces_list[0].shape
    img_size = reduce(operator.mul, img_shape)
    nb_images = len(eigenfaces_list)
    eigenfaces_list = np.array(eigenfaces_list).reshape(nb_images, img_size)

    # Scale images between 0 and 1
    def normalize_image(image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    eigenfaces_normalised_list = [normalize_image(img) for img in eigenfaces_list]
    #print(eg1_norm.min(), eg1_norm.max())
    if show_img:
        eg1_norm = eigenfaces_normalised_list[show_img]
        print(img_size)
        image_numpy_to_pillow(eg1_norm, img_shape).show()


    # Apply Laplace
    def add_laplace_noise(image, epsilon):
        # Apply laplace noise with delta_f= 1
        # $$ \frac{\epsilon}{2 \triangle f} e^{- \frac{\|x-FSV_i\| \epsilon}{\triangle F}} $$
        scale = 1 / epsilon  # self.sensitivity[subject] / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=image.shape)
        return image + noise
    noised_image_list = [add_laplace_noise(img, epsilon) for img in eigenfaces_normalised_list]
    if show_img:
        eg1_noised = noised_image_list[show_img]
        image_numpy_to_pillow(eg1_noised, img_shape).show()

    if export_img:
        # transform all images in pil
        pil_images_source = [image.resize(img_shape) for image in image_df['userFaces'].tolist()]
        pil_eigenface_images = [image_numpy_to_pillow(img, img_shape) for img in eigenfaces_list]
        pil_normalised_images = [image_numpy_to_pillow(img, img_shape) for img in eigenfaces_normalised_list]
        pil_noised_images = [image_numpy_to_pillow(img, img_shape) for img in noised_image_list]
        # Create the united image
        image_lists = [pil_images_source, pil_eigenface_images, pil_normalised_images, pil_noised_images]
        result = combine_pill_images_lists(image_lists, delimiter_color=(100, 100, 100), delimiter_width=10)
        # save
        result.save(f"{export_img_folder}/subject_{subject}-noised_image_with_epsilon_{epsilon}.png")

    # return noised images
    return np.array(noised_image_list), peep


def generate_noise_on_projected_data(subject=1, epsilon=5):
    # Import images
    image_df = import_subject_images(subject)

    # Generate Projected_data
    peep = Peep()
    peep.run_from_dataframe(image_df, epsilon)
    projected_vectors = peep.get_projected_data()[15]
    #print(projected_vectors.shape, projected_vectors[0])

    # Scale images between 0 and 1
    def normalize_vector(vector):
        return (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    normalised_projected_vectors = np.array([normalize_vector(vec) for vec in projected_vectors])
    #print(normalised_projected_vectors.shape, normalised_projected_vectors[0])

    # Apply Laplace
    def add_laplace_noise(vector, epsilon):
        # Apply laplace noise with delta_f= 1
        # $$ \frac{\epsilon}{2 \triangle f} e^{- \frac{\|x-FSV_i\| \epsilon}{\triangle F}} $$
        scale = 1 / epsilon  # self.sensitivity[subject] / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=vector.shape)
        return vector + noise
    noised_vectors = np.array([add_laplace_noise(vec, epsilon) for vec in normalised_projected_vectors])
    #print(noised_vectors.shape, noised_vectors[0])

    # Return noised vectors
    return noised_vectors, peep



def wf_pil_image():
    for e in np.arange(0.5, 9.5, 0.25):
        generate_noise_on_eigenface_images(subject=1, epsilon=e, show_img=False, export_img=True)


def wf_pil_vector():
    folder_save = "../../sandbox/noised_image_test"
    e_pil_list = []
    for e in np.arange(0.00001, 0.3001, 0.05):
        print(e)
        noised_vectors, peep = generate_noise_on_projected_data(subject=1, epsilon=e)
        pca = peep.pca_objects[15].pca
        reconstructed_data = pca.inverse_transform(noised_vectors)
        pil_noised_images = [image_numpy_to_pillow(img, (100,100)) for img in reconstructed_data]
        e_pil_list += [pil_noised_images]
    result = combine_pill_images_lists(e_pil_list, delimiter_color=(100, 100, 100), delimiter_width=10)
    result.save(f"{folder_save}/result_from_vectors.png")



if __name__ == '__main__':

    #wf_pil_image()
    #wf_pil_vector()
    #noised_vectors, peep = generate_noise_on_projected_data(subject=1, epsilon=4)

    noised_images, peep = generate_noise_on_eigenface_images(subject=1, epsilon=4, show_img=False, export_img=True)

    noiseImg = noised_images[3]
    pca = peep.pca_objects[15].pca
    #image_numpy_to_pillow(noiseImg, (100, 100)).show()

    print(noised_images.shape)
    print(pca.components_.shape)

    reconstructed_noisy = pca.inverse_transform(noiseImg)

    ######################################################""
    # A partir des noised images, regenerate les vecteurs avec une nouvelle pca pour les donner à pca.inverse_transform()
    # Ca devrait fonctionner normalement :/
    ######################################################""

    # Test qui fonctionne pas
    # #!!a partir d'une image
    #noiseImg = np.array(noised_image_list)
    #noiseImg = noiseImg.reshape(10, 10000)
    #print(noiseImg)
    #reconstructed_noisy = pca.inverse_transform(noiseImg)

    # inverse_transform qui fonctionne avec peep.projected_data
    # !!a partir des pca.vecteurs
    #projected_data = peep.get_projected_data()
    #for subject, projected_images in projected_data.items():
    #    print(projected_images.shape)
    #    print(projected_images)
    #    reconstructed_noisy = pca.inverse_transform(projected_images)

    #img_name = f"{epsilon}.1-eigenface_{epsilon}"
    #export_images(eigenfaces_list, folder_save, img_name, "eigenface", True)
