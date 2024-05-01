import xml.dom.minidom
import json
import cv2
import os


def get_file_list(root_dir, ext_filter=None):
    """
        get all files in root_dir directory
    """
    path_list = []
    file_list = []
    join_list = []
    for path, _, files in os.walk(root_dir):
        for name in files:
            if ext_filter is None:
                pass
            elif any(name.lower().endswith(s.lower()) for s in ext_filter):
                pass
            else:
                continue

            path_list.append(path)
            file_list.append(name)
            join_list.append(os.path.join(path, name))

    return path_list, file_list, join_list


def get_info_annotation_xml(xml_file):
    dom = xml.dom.minidom.parse(xml_file)  # or xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = dom.toprettyxml()
    xml_lines = pretty_xml_as_string.splitlines()
    info_filename = ''
    info_width = ''
    info_height = ''
    info_objects = []
    info_object_name = ''
    info_object_xmin = ''
    info_object_xmax = ''
    info_object_ymin = ''
    info_object_ymax = ''

    for i in range(len(xml_lines)):
        line = xml_lines[i]

        if '<filename>' in line and '</filename>' in line:
            info_filename = line[line.index('<filename>') + 10:line.index('</filename>')]
        elif '<width>' in line and '</width>' in line:
            info_width = int(line[line.index('<width>') + 7:line.index('</width>')])
        elif '<height>' in line and '</height>' in line:
            info_height = int(line[line.index('<height>') + 8:line.index('</height>')])
        elif '<name>' in line and '</name>' in line:
            info_object_name = line[line.index('<name>') + 6:line.index('</name>')]
        elif '<xmin>' in line and '</xmin>' in line:
            info_object_xmin = float(line[line.index('<xmin>') + 6:line.index('</xmin>')])
        elif '<xmax>' in line and '</xmax>' in line:
            info_object_xmax = float(line[line.index('<xmax>') + 6:line.index('</xmax>')])
        elif '<ymin>' in line and '</ymin>' in line:
            info_object_ymin = float(line[line.index('<ymin>') + 6:line.index('</ymin>')])
        elif '<ymax>' in line and '</ymax>' in line:
            info_object_ymax = float(line[line.index('<ymax>') + 6:line.index('</ymax>')])
        elif '</bndbox>' in line:
            info_objects.append({'name': info_object_name,
                                 'box': [info_object_xmin, info_object_ymin, info_object_xmax, info_object_ymax]})

    return {'filename': info_filename, 'width': info_width, 'height': info_height, 'objects': info_objects}


def display_result(img, annotation, dataset_type='pascal'):
    img_h, img_w = img.shape[:2]

    if dataset_type == 'pascal':
        ann_h, ann_w = annotation['height'], annotation['width']
        ann_object = annotation['objects']

        if img_w == ann_w and img_h == ann_h:
            print('Image size is matched!')
        else:
            print("Warning!, Image size isn't matched!")

        for i in range(len(ann_object)):
            obj_name = ann_object[i]['name']
            [x1, y1, x2, y2] = ann_object[i]['box']

            if obj_name == 'mask':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, obj_name, (int(x1) + 5, int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    else:   # 'yolo'
        for label_line in annotation.splitlines():
            [label_id, cx, cy, sw, sh] = label_line.split()
            x1 = int((float(cx) - float(sw) / 2) * img_w)
            x2 = int((float(cx) + float(sw) / 2) * img_w)
            y1 = int((float(cy) - float(sh) / 2) * img_h)
            y2 = int((float(cy) + float(sh) / 2) * img_h)
            color = (0, 0, 255)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label_id, (int(x1) + 5, int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def generate_xml(save_path, annotation):
    filename = annotation['filename']
    width = annotation['width']
    height = annotation['height']
    objects = annotation['objects']

    part1 = """<annotation verified="yes">
    <folder>Annotation</folder>
    <filename>{}</filename>
    <path>/JPEGImages/{}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>""".format(filename, filename, width, height)

    for i in range(len(objects)):
        obj_name = objects[i]['name']
        [x1, y1, x2, y2] = objects[i]['box']
        part2 = """    <object>
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>""".format(obj_name, x1, y1, x2, y2)

        part1 += '\n' + part2

    part1 += '\n</annotation>'

    f = open(save_path, "w")
    f.write(part1)
    f.close()


def convert_yolo_to_xml_object(image_file, yolo_label_file, object_list):
    img = cv2.imread(image_file)
    img_h, img_w = img.shape[:2]
    annotation = read_text(yolo_label_file)
    label_list = annotation.splitlines()

    info_objects = []
    for i in range(len(label_list)):
        word_list = label_list[i].split()
        cx = float(word_list[1]) * img_w
        cy = float(word_list[2]) * img_h
        w = float(word_list[3]) * img_w
        h = float(word_list[4]) * img_h
        info_objects.append({'name': object_list[int(word_list[0])],
                             'box': [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]})

    xml_object = {'filename': yolo_label_file.split('/')[-1].split('.')[0],
                  'width': img_w,
                  'height': img_h,
                  'objects': info_objects}

    return xml_object


def scale_annotation(img, annotation, scale):
    img = cv2.resize(img, None, fx=scale, fy=scale)

    annotation['width'] = int(annotation['width'] * scale)
    annotation['height'] = int(annotation['height'] * scale)

    for obj_ind in range(len(annotation['objects'])):
        [xmin, ymin, xmax, ymax] = annotation['objects'][obj_ind]['box']
        annotation['objects'][obj_ind]['box'] = \
            [int(xmin * scale), int(ymin * scale), int(xmax * scale), int(ymax * scale)]

    return img, annotation


def read_text(filename):
    with open(filename, 'r') as fid:
        text = fid.read()

    return text


def write_text(filename, text):
    with open(filename, 'w') as fid:
        fid.write(text)


def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    return data


def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def split_enpro_image(img, left_val=0.6, right_val=0.35):
    img_h, img_w = img.shape[:2]
    img_left = img.copy()
    img_right = img.copy()
    cv2.rectangle(img_left, (int(img_w * left_val), 0), (img_w, img_h), (0, 0, 0), -1)
    cv2.rectangle(img_right, (0, 0), (int(img_w * right_val), img_h), (0, 0, 0), -1)
    return img_left, img_right
