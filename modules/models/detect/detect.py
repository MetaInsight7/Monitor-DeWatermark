import os
import cv2
import numpy as np
import onnxruntime as ort
from utils.img_tools import LetterBox


class YOLOv8():
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, config):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
      
        self.onnx_model = config.detect.model_path
        self.conf_thres = config.detect.conf_thres
        self.iou_thres = config.detect.iou_thres
        self.output_folder = os.path.join(config.output_folder,"detect")
        self.save_result = config.detect.save_result 
        self.session  = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Load the class names from the COCO dataset
        self.classes = {0: 'time', 1: 'place', 2: 'task'}

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Read the input image using OpenCV
        self.img = cv2.imread(img)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)


        # resize img
        letterbox = LetterBox(new_shape=[self.input_width, self.input_height], auto=False, stride=32)
        img= letterbox(image=img)

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data


    def postprocess(self, img_path, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.conf_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x1, y1, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                x1 = x1 - w / 2
                y1 = y1 - h / 2
   
                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([x1, y1, w, h])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)

        # Calculate the scaling factors for the bounding box coordinates
        gain = min(self.input_width / self.img_width, self.input_height / self.img_height)
        pad = (
            round((self.input_width - self.img_width * gain) / 2 - 0.1),
            round((self.input_height - self.img_height * gain) / 2 - 0.1),
        )

        # Iterate over the selected indices after non-maximum suppression
        res = {'img_name':img_path, 'box':[], 'score':[], 'class_id':[]}
        
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            box[0] = int((box[0] - pad[0]) / gain)
            box[1] = int((box[1] - pad[1]) / gain)
            box[2] = int(box[2] / gain)
            box[3] = int(box[3] / gain)

            res['box'].append(box)
            res['score'].append(score)
            res['class_id'].append(class_id)

            # Draw the detection on the input image
            if self.save_result:
                self.draw_detections(self.img, box, score, class_id)

        # Return the modified input image
        return res, self.img


    def __call__(self, img_path):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers

        # Get the model inputs
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data = self.preprocess(img_path)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        detect_res, output_img  = self.postprocess(img_path, outputs)  # output image
        if self.save_result:
            os.makedirs(self.output_folder, exist_ok=True)
            cv2.imwrite(os.path.join(self.output_folder, os.path.basename(img_path)), output_img)
        return detect_res