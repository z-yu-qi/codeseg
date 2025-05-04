import xml.etree.ElementTree as ET
import cv2
import os
import yaml

# 从config.yaml加载故障类型
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    CLASSES = list(config['fault_types'].keys())

# 定义原始标注类别到新故障类型的映射
CLASS_MAPPING = {
    "defect": "insulator_damage",  # 将defect映射到绝缘子损坏
    "insulator": "insulator_damage",  # 将insulator映射到绝缘子损坏
    "cable": "wire_wear",  # 将cable映射到导线磨损
    "tower_lattice": "corrosion",  # 将tower_lattice映射到金具腐蚀
    "bird_nest": "foreign_object"  # 将bird_nest映射到异物附着
}

def parse_xml(xml_path):
    """
    解析XML标注文件，提取边界框和类别信息
    返回格式：
        {
            "filename": "image.jpg",
            "width": 640,
            "height": 480,
            "objects": [
                {"name": "cable", "bbox": [xmin, ymin, xmax, ymax]},
                {"name": "insulator", "bbox": [...]}
            ]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {
        "filename": root.find("filename").text,
        "width": int(root.find("size/width").text),
        "height": int(root.find("size/height").text),
        "objects": []
    }

    for obj in root.iter("object"):
        try:
            obj_name = obj.find("name").text
        except:
            try:
                obj_name = obj.find("n").text
            except:
                print(f"警告：在文件 {xml_path} 中找不到对象名称标签")
                continue

        bbox = obj.find("bndbox")
        bbox_coords = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text)
        ]

        # 使用映射字典转换类别名称
        mapped_name = CLASS_MAPPING.get(obj_name, CLASSES[0])  # 如果找不到映射，使用第一个故障类型
        if mapped_name not in CLASSES:
            print(f"警告：类别 {obj_name} 映射到 {mapped_name}，但该类别不在配置文件中")
            mapped_name = CLASSES[0]
            
        data["objects"].append({"name": mapped_name, "bbox": bbox_coords})

    return data


def xml_to_yolo(xml_dir, output_dir, class_names):
    """
    将XML标注转换为YOLO格式（每个图像对应一个TXT文件）
    class_names: 类别列表，如 ["cable", "insulator"]
    """
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        data = parse_xml(xml_path)

        # 生成YOLO格式的TXT文件
        txt_filename = os.path.splitext(xml_file)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, "w") as f:
            for obj in data["objects"]:
                class_id = class_names.index(obj["name"])
                x_center = (obj["bbox"][0] + obj["bbox"][2]) / 2 / data["width"]
                y_center = (obj["bbox"][1] + obj["bbox"][3]) / 2 / data["height"]
                width = (obj["bbox"][2] - obj["bbox"][0]) / data["width"]
                height = (obj["bbox"][3] - obj["bbox"][1]) / data["height"]

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"转换完成！结果保存在 {output_dir}")


def visualize_xml_annotations(image_dir, xml_dir):
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        data = parse_xml(xml_path)
        image_path = os.path.join(image_dir, data["filename"])

        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        for obj in data["objects"]:
            xmin, ymin, xmax, ymax = obj["bbox"]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, obj["name"], (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Annotation", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_data_yaml(xml_dir, output_path, class_names):
    """
    根据XML文件自动生成YOLO格式的data.yaml
    """
    data = {
        "train": "../train/images",
        "val": "../val/images",
        "nc": len(class_names),
        "names": class_names
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f)

    print(f"生成data.yaml：{output_path}")

if __name__ == "__main__":
    # 步骤1：转换XML到YOLO格式
    xml_to_yolo(
        xml_dir="datasets/data/train/yuan",
        output_dir="datasets/data/train/labels",
        class_names=CLASSES
    )

    # 步骤2：生成data.yaml
    generate_data_yaml(
        xml_dir="datasets/data/train/yuan",
        output_path="datasets/data.yaml",
        class_names=CLASSES
    )

    # 步骤3：可视化验证（可选）
    visualize_xml_annotations(
        image_dir="datasets/images",
        xml_dir="datasets/data/train/yuan"
    )