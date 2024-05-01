import engine , json
import numpy as np
import random
import shutil
import cv2
import sys
import os


def enhance_data(img_file, label_file, en_noise=False, en_denoise=False, en_rotate=False, num_rotate=-1,
                 en_bright=False, en_contrast=False, debug=False, dataset_type='yolo'):
    image = cv2.imread(img_file)
    scale = 0.5

    if debug:
        cv2.imshow('org', cv2.resize(image, None, fx=scale, fy=scale))
        cv2.waitKey()

    img_ret = image.copy()
    if en_noise:
        print('en_noise',en_noise)
        row, col, ch = img_ret.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = 20 * np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        gauss = gauss.astype(np.uint8)
        img_ret = image + gauss

    if en_denoise:
        print('en_denoise',en_denoise)
        img_ret = cv2.fastNlMeansDenoisingColored(img_ret, None, 5, 5, 7, 21)

    if en_bright:
        print('en_bright',en_bright)
        brightness = random.randint(-60, 60)
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        img_ret = cv2.addWeighted(img_ret, alpha_b, img_ret, 0, gamma_b)

    if en_contrast:
        print('en_contrast',en_contrast)
        contrast = random.randint(-30, 25)
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        img_ret = cv2.addWeighted(img_ret, alpha_c, img_ret, 0, gamma_c)

    if dataset_type == 'pascal':
        annotation = engine.get_info_annotation_xml(label_file)
    else:   # 'yolo'
        annotation = engine.read_text(label_file)

    if en_rotate:
        if num_rotate == -1:
            rot = random.randint(0, 2)
        else:
            rot = num_rotate
        for _ in range(rot + 1):
            img_ret = np.rot90(img_ret)

            if dataset_type == 'pascal':
                h, w = annotation['height'], annotation['width']
                annotation['width'], annotation['height'] = h, w
                for i in range(len(annotation['objects'])):
                    [x1, y1, x2, y2] = annotation['objects'][i]['box']
                    annotation['objects'][i]['box'] = [y1, w - x2, y2, w - x1]
            else:   # 'yolo'
                new_annotation = []
                for label_line in annotation.splitlines():
                    [label_id, pos_cx, pos_cy, size_w, size_h] = label_line.split()
                    new_annotation.append(' '.join([label_id, pos_cy, str(1 - float(pos_cx)), size_h, size_w]))

                annotation = '\n'.join(new_annotation)

    if debug:
        cv2.imshow('ret', cv2.resize(img_ret, None, fx=scale, fy=scale))
        cv2.waitKey()

    return img_ret, annotation


def process_dataset_pascal(img_dataset, label_dataset, out_img, out_xml,augmentations):
    _, _, img_list = engine.get_file_list(img_dataset, ext_filter=['jpg'])
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_xml, exist_ok=True)
    new_items = []
    for i in range(len(img_list)):
        if (i + 1) % 10 == 0:
            print(f'{i + 1}/{len(img_list)}')

        base_name = img_list[i].split('/')[-1].split('.')[0]
        img_file = os.path.join(img_dataset, base_name + '.jpg')
        xml_file = os.path.join(label_dataset, base_name + '.xml')
        print(augmentations['noise'])
        img_new, xml_new = enhance_data(img_file, xml_file, en_noise=augmentations['noise'], dataset_type='pascal')
        cv2.imwrite(os.path.join(out_img, base_name + '_1.jpg'), img_new)
        xml_new['filename'] = xml_new['filename'] + '_1'
        engine.generate_xml(os.path.join(out_xml, base_name + '_1.xml'), xml_new)
        new_items.append(base_name + '_1')

        img_new, xml_new = enhance_data(img_file, xml_file, en_denoise=augmentations['denoise'], dataset_type='pascal')
        cv2.imwrite(os.path.join(out_img, base_name + '_2.jpg'), img_new)
        xml_new['filename'] = xml_new['filename'] + '_2'
        engine.generate_xml(os.path.join(out_xml, base_name + '_2.xml'), xml_new)
        new_items.append(base_name + '_2')

        img_new, xml_new = enhance_data(img_file, xml_file, en_contrast=augmentations['contrast'], dataset_type='pascal')
        cv2.imwrite(os.path.join(out_img, base_name + '_3.jpg'), img_new)
        xml_new['filename'] = xml_new['filename'] + '_3'
        engine.generate_xml(os.path.join(out_xml, base_name + '_3.xml'), xml_new)
        new_items.append(base_name + '_3')

        img_new, xml_new = enhance_data(img_file, xml_file, en_bright=augmentations['brightness'], dataset_type='pascal')
        cv2.imwrite(os.path.join(out_img, base_name + '_4.jpg'), img_new)
        xml_new['filename'] = xml_new['filename'] + '_4'
        engine.generate_xml(os.path.join(out_xml, base_name + '_4.xml'), xml_new)
        new_items.append(base_name + '_4')
        
        img_new, xml_new = enhance_data(img_file, xml_file, en_rotate=augmentations['rotation'], dataset_type='pascal')
        cv2.imwrite(os.path.join(out_img, base_name + '_5.jpg'), img_new)
        xml_new['filename'] = xml_new['filename'] + '_5'
        engine.generate_xml(os.path.join(out_xml, base_name + '_5.xml'), xml_new)
        new_items.append(base_name + '_5')

    engine.write_text('../trainval_enhance.txt', '\n'.join(new_items))


def process_dataset_yolo(img_dataset, label_dataset, out_img, out_label,augmentations):
    print(type(augmentations))
    _, _, img_list = engine.get_file_list(img_dataset, ext_filter=['jpg'])
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_label, exist_ok=True)
    for i in range(len(img_list)):
        if (i + 1) % 10 == 0:
            print(f'{i + 1}/{len(img_list)}')
        base_name = img_list[i].split('/')[-1].split('.')[0]
        img_file = os.path.join(img_dataset, base_name + '.jpg')
        label_file = os.path.join(label_dataset, base_name + '.txt')

        # Ignore empty label
        annotation = engine.read_text(label_file)
        if len(annotation) < 2:
            continue

        shutil.copy(img_file, os.path.join(out_img, base_name + '_0.jpg'))
        shutil.copy(label_file, os.path.join(out_label, base_name + '_0.txt'))
        print("Augmentations dictionary:", augmentations)
        print("Type of augmentations:", type(augmentations))
        print("Value of augmentations['noise']:", augmentations['noise'])
        

        img_new, label_new = enhance_data(img_file, label_file, en_noise=augmentations['noise'], dataset_type='yolo')
        cv2.imwrite(os.path.join(out_img, base_name + '_1.jpg'), img_new)
        engine.write_text(os.path.join(out_label, base_name + '_1.txt'), label_new)

        img_new, label_new = enhance_data(img_file, label_file, en_denoise=augmentations['denoise'], dataset_type='yolo')
        cv2.imwrite(os.path.join(out_img, base_name + '_2.jpg'), img_new)
        engine.write_text(os.path.join(out_label, base_name + '_2.txt'), label_new)

        img_new, label_new = enhance_data(img_file, label_file, en_contrast=augmentations['contrast'], dataset_type='yolo')
        cv2.imwrite(os.path.join(out_img, base_name + '_3.jpg'), img_new)
        engine.write_text(os.path.join(out_label, base_name + '_3.txt'), label_new)

        img_new, label_new = enhance_data(img_file, label_file, en_bright=augmentations['brightness'], dataset_type='yolo')
        cv2.imwrite(os.path.join(out_img, base_name + '_4.jpg'), img_new)
        engine.write_text(os.path.join(out_label, base_name + '_4.txt'), label_new)
        
        #img_new, xml_new = enhance_data(img_file, xml_file, en_rotate=augmentations['rotation'], dataset_type='pascal')
        #cv2.imwrite(os.path.join(out_img, base_name + '_5.jpg'), img_new)
        #xml_new['filename'] = xml_new['filename'] + '_5'
        #engine.generate_xml(os.path.join(out_xml, base_name + '_5.xml'), xml_new)
        #new_items.append(base_name + '_5')


if __name__ == '__main__':
    # f_img = '../result_image/00000.jpg'
    # f_xml = '../result_annotation/scai/obj_train_data/00000.txt'

    # ret_img, ret_annot = enhance_data(f_img, f_xml,
    #                                   en_noise=False,
    #                                   en_denoise=False,
    #                                   en_bright=False,
    #                                   en_contrast=False,
    #                                   en_rotate=True,
    #                                   dataset_type='yolo',
    #                                   debug=True)

    # cv2.imwrite('ret.jpg', ret_img)
    # # engine.generate_xml('ret.xml', ret_annot)
    # engine.write_text('ret.txt', ret_annot)
    #
    # img_final = engine.display_result(cv2.imread('ret.jpg'), ret_annot, dataset_type='yolo')
    # cv2.imshow('d', cv2.resize(img_final, None, fx=0.5, fy=0.5))
    # cv2.waitKey()

    input_aug_list = ['yolo', 'dataset_enhance_config.json']

    for arg_ind in range(len(sys.argv) - 1):
        input_aug_list[arg_ind] = sys.argv[arg_ind + 1]

    dataset_format = input_aug_list[0]
    config_path = input_aug_list[1]

    config = engine.read_json(config_path)
    augmentations = config.get("augmentations", {})
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Extract the augmentations dictionary from the config_data
        augmentations = config_data.get("augmentations", {})
    if dataset_format == 'yolo':
        process_dataset_yolo(img_dataset=config["original_image_folder"],
                             label_dataset=config["original_label_folder"],
                             out_img=config["enhance_image_folder"],
                             out_label=config["enhance_label_folder"],
                             augmentations=augmentations)
    else:
        process_dataset_pascal(img_dataset=config["original_image_folder"],
                               label_dataset=config["original_label_folder"],
                               out_img=config["enhance_image_folder"],
                               out_xml=config["enhance_label_folder"],
                               augmentations=augmentations)
