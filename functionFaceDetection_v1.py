from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
from google.cloud import storage

def facedetection(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    storage_client = storage.Client()
    vision_client = vision.ImageAnnotatorClient()

    in_bucket_name = 'team-img'
    out_bucket_name = 'team-out'
    
    in_bucket = storage_client.get_bucket(in_bucket_name)
    in_blob = in_bucket.blob(event['name'])
    
    temp_filename = str(f"/tmp/{event['name']}.tmp")
    content = None
    with open(temp_filename, 'wb') as in_file:
        in_blob.download_to_file(in_file)
        with open(temp_filename, 'rb') as r_file:
            content = r_file.read()

    image = types.Image(content=content)
    faces = vision_client.face_detection(image=image).face_annotations
    
    max = -1
    selected = None
    for face in faces:
        if max < face.sorrow_likelihood:
            max = face.sorrow_likelihood
            selected = face
    if max == -1:
        print(f"face detection failed for {event['mediaLink']}")
        raise Exception("face detection failed!")

    im = Image.open(temp_filename)
    draw = ImageDraw.Draw(im)
    box = [(vertex.x, vertex.y) for vertex in selected.bounding_poly.vertices]
    draw.line(box + [box[0]], width=5, fill='#00ff00')
    highlighted_filename = str(f"/tmp/highlighted_{event['name']}")
    im.save(highlighted_filename)

    out_bucket = storage_client.get_bucket(out_bucket_name)
    destination_blob_name = str(f"mod_{event['name']}")
    out_blob = out_bucket.blob(destination_blob_name)
    out_blob.upload_from_filename(highlighted_filename)