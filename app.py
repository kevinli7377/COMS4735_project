from flask import Flask, render_template, request, abort, send_from_directory
import os
from system import app_main
import shutil
import cv2 as cv

app = Flask(__name__)

#-------------------Application configuration------------------------
#--------------------------------------------------------------------
app.config['STATIC'] = 'static/'
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['STATIC'], 'upload/')
app.config['REF_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'references/')
app.config['QUERY_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'query/')

app.config['BINARY'] = os.path.join(app.config['STATIC'],'binary/')
app.config['REF_BIN'] = os.path.join(app.config['BINARY'],'references_binary/')
app.config['QUERY_BIN'] = os.path.join(app.config['BINARY'],'queries_binary/')

#-------------------Application helper methods-----------------------
#--------------------------------------------------------------------
def delete_files_in_directory(directory):
    '''Delete all preexisting files in a given directory'''
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def get_image_paths(folder):
    '''Gets paths of all images in folder'''
    image_extensions = ('.jpg', '.jpeg', '.png')
    filenames = os.listdir(folder)
    image_paths = [os.path.join(folder, filename) for filename in filenames if filename.lower().endswith(image_extensions)]
    return image_paths

def is_valid_image(file):
    '''Check if uploaded file is valid'''
    allowed_mimetypes = {'image/jpeg', 'image/png', 'image/jpg'}
    return file.mimetype in allowed_mimetypes

#-------------------Main application methods-------------------------
#--------------------------------------------------------------------

@app.route('/')
def index():
    # Render template
    return render_template('index.html')

@app.route('/binary/<path:path>')
def serve_binary_images(path):
    return send_from_directory('static/binary', path)

@app.route('/submit', methods=['POST'])
def submit():
    '''
    Submission handler:
    1. Intializaiton - removal of files from preexisting system run
    2. File and input retrieval from front-end
    3. main() to estimate volume in query
    4. Results processing
    5. Rendering results.html page
    '''
    if request.method == 'POST':

        # 1. Initialization - create directories if they do not exist

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        if not os.path.exists(app.config['REF_FOLDER']):
            os.makedirs(app.config['REF_FOLDER'])

        if not os.path.exists(app.config['QUERY_FOLDER']):
            os.makedirs(app.config['QUERY_FOLDER'])

        if not os.path.exists(app.config['BINARY']):
            os.makedirs(app.config['BINARY'])

        if not os.path.exists(app.config['REF_BIN']):
            os.makedirs(app.config['REF_BIN'])

        if not os.path.exists(app.config['QUERY_BIN']):
            os.makedirs(app.config['QUERY_BIN'])

        # Deleting files from previous run
        delete_files_in_directory(app.config['REF_FOLDER'])
        delete_files_in_directory(app.config['QUERY_FOLDER'])
        delete_files_in_directory(app.config['REF_BIN'])
        delete_files_in_directory(app.config['QUERY_BIN'])

        # 2. Retrieve files from front-end submission and store

        ref_images = request.files.getlist('ref_image[]')  # Use getlist() to handle multiple files
        ref_volumes = request.form.getlist('ref_volume[]')  # Use getlist() to handle multiple volumes

        query_image = request.files['query_image']
        query_volume = request.form['query_volume']

        # Convert all numerical inputs for ref_images and check if numerical inputs are valid
        for i in range(len(ref_volumes)):
            try:
                ref_volumes[i] = float(ref_volumes[i])
                query_volume = float(query_volume)
            except ValueError:
                abort(400, description="Invalid reference/query volume (ml) value. It should be a float or an integer.")

        # Checking if reference file uploads are valid. If so, store as img, volume pairs
        image_volume_pairs = []
        for ref_image, ref_volume in zip(ref_images, ref_volumes):

            if not is_valid_image(ref_image):
                abort(400, description="Invalid reference image format.")

            with open(os.path.join(app.config['REF_FOLDER'], ref_image.filename), 'wb') as f:
                shutil.copyfileobj(ref_image.stream, f)

            image_path = os.path.join(app.config['REF_FOLDER'], ref_image.filename)
            image_volume_pairs.append((image_path, ref_volume))

        if not is_valid_image(query_image):
            abort(400, description="Invalid query image format.")

        with open(os.path.join(app.config['QUERY_FOLDER'], query_image.filename), 'wb') as f:
            shutil.copyfileobj(query_image.stream, f)

        query_image_path = get_image_paths(app.config['QUERY_FOLDER'])[0]

        ref_images_paths = [path for path,volume in image_volume_pairs]
        ref_volumes = [volume for path ,volume in image_volume_pairs]

        # 3. Calling main system
        ref_liquid_images_list, q_liquid_image, estimation, relative_diff, abs_diff = app_main(ref_images_paths,ref_volumes,query_image_path,query_volume)

        # 4. Results processing

        # Storing reference images
        for i, (b_img, volume) in enumerate(zip(ref_liquid_images_list, ref_volumes)):
            output_path = os.path.join(app.config['REF_BIN'],f'ref_liquid_image_{i}.jpg')
            cv.imwrite(output_path,b_img)

        # Storing query image
        q_output_path = os.path.join(app.config['QUERY_BIN'],f'q_liquid_image.jpg')
        cv.imwrite(q_output_path,q_liquid_image)

        # Get image paths
        ref_bimages_paths = [os.path.join(app.config['REF_BIN'], f'ref_liquid_image_{i}.jpg') for i in range(len(ref_volumes))]
        query_bimage_path = get_image_paths(app.config['QUERY_BIN'])[0]

        # Convert absolute paths to relative paths
        ref_bimages_paths = [os.path.relpath(path, app.config['STATIC']) for path in ref_bimages_paths]
        query_bimage_path = os.path.relpath(query_bimage_path, app.config['STATIC'])

        # 5. Rendering results.html
        return render_template('results.html', ref_liquid_images=ref_bimages_paths, q_liquid_image=query_bimage_path, 
                               estimation=estimation, relative_diff=relative_diff, abs_diff=abs_diff,
                               ref_volumes = ref_volumes,query_volume=query_volume)

if __name__ == '__main__':
    app.run(debug=True)