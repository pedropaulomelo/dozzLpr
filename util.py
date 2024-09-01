import string
import easyocr
import os
import cv2  # Adicione esta linha para importar o OpenCV
from datetime import datetime

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'Z': '2',
                    'J': '3',
                    'A': '4',
                    'S': '5',
                    'G': '6',                    
                    'B': '8'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '2': 'Z',
                    '3': 'J',
                    '4': 'A',
                    '5': 'S',
                    '6': 'G',
                    '8': 'B'}

# Define the global output path
output_csv_path = 'results.csv'
plates_directory = 'plates'

# Ensure the directory exists
if not os.path.exists(plates_directory):
    os.makedirs(plates_directory)

def initialize_csv():
    """
    Initialize the CSV file by creating the header if it doesn't exist.
    """
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w') as f:
            f.write('timestamp,license_number,license_number_score,image_path\n')
        f.close()

def write_csv(result, image_path):
    """
    Append the results to the CSV file for a single image.

    Args:
        result (dict): Dictionary containing the results.
        image_path (str): Path to the saved image.
    """
    with open(output_csv_path, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Criar o link clicável para a imagem
        image_link = f'{image_path}'
        f.write('{}, {}, {}, {}\n'.format(
            timestamp,
            result['license_plate']['text'],
            result['license_plate']['text_score'],
            image_link)
        )
    f.close()

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()):
        return True
    else:
        return False

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = { 0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char, 3: dict_char_to_int, 4: dict_int_to_char, 5: dict_char_to_int, 6: dict_char_to_int }
    for j in range(7):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image and log the result.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        
        text = text.upper().replace(' ', '')
        result = {
            'license_plate': {
                'text': text,
                'text_score': score,
            }
        }
        if score > 0.2 and license_complies_format(text):
            print('LICENSE COMPLIES AND CONFIDENCE IS HIGH')
            result['license_plate']['text'] = format_license(text)
            
            # Save the license plate image
            license_plate_image_path = os.path.join(plates_directory, f"{result['license_plate']['text']}.jpg")
            cv2.imwrite(license_plate_image_path, license_plate_crop)  # Use cv2.imwrite to save the image

            # Write the result and image path to the CSV
            write_csv(result, license_plate_image_path)

            return result['license_plate']['text'], result['license_plate']['text_score']
    print('LICENSE DOESN\'T COMPLY OR CONFIDENCE TOO LOW')
    return None, None

def get_car(license_plate_bbox, vehicle_bboxes):
    """
    Find the closest vehicle to the given license plate.

    Args:
        license_plate_bbox (list): Bounding box of the license plate [x1, y1, x2, y2].
        vehicle_bboxes (list): List of vehicle bounding boxes [[x1, y1, x2, y2, ...], ...].

    Returns:
        list: Closest vehicle bounding box [x1, y1, x2, y2].
    """
    lx1, ly1, lx2, ly2 = license_plate_bbox
    closest_vehicle_bbox = None
    min_distance = float('inf')

    for vehicle_bbox in vehicle_bboxes:
        vx1, vy1, vx2, vy2 = vehicle_bbox[:4]  # Extract only the bounding box coordinates
        # Calculate the center point of vehicle and license plate bboxes
        v_center = ((vx1 + vx2) / 2, (vy1 + vy2) / 2)
        l_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
        # Calculate Euclidean distance between centers
        distance = ((v_center[0] - l_center[0]) ** 2 + (v_center[1] - l_center[1]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_vehicle_bbox = vehicle_bbox[:4]

    return closest_vehicle_bbox

# Initialize the CSV file
initialize_csv()