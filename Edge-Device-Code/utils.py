import numpy as np
import cv2
import boto3
import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("parking_utils")

def decode_yolov8(output_layer, confidence_threshold=0.3, iou_threshold=0.5):
    """
    Decodes raw YOLOv8 output layer data from the DepthAI VPU.
    
    Args:
        output_layer: Raw output array from neural network node.
        confidence_threshold: Minimum confidence score to retain detection.
        iou_threshold: Overlap threshold for Non-Maximum Suppression (NMS).
        
    Returns:
        list: Normalized bounding boxes [[x_min, y_min, w, h], ...] scaling 0.0 to 1.0.
    """
    # YOLOv8 output layout: (1, 4 + num_classes, 8400) -> e.g., (1, 6, 8400) for 2 classes
    try:
        data = np.array(output_layer).reshape(6, 8400).transpose()
    except Exception as e:
        logger.error(f"Failed to reshape output layer. Shape might be different: {e}")
        return []
    
    # Class 1 is 'space-occupied' (Index 5 because: 0-3 = x,y,w,h, 4 = empty, 5 = occupied)
    scores = data[:, 5]
    mask = scores > confidence_threshold
    if not np.any(mask):
        return []
        
    filtered_data = data[mask]
    filtered_scores = scores[mask]
    boxes = filtered_data[:, :4]
    
    # Convert center coordinates (cx, cy, w, h) to top-left coordinates (x, y, w, h)
    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)
    
    # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(), 
        scores=filtered_scores.tolist(), 
        score_threshold=confidence_threshold, 
        nms_threshold=iou_threshold
    )
    
    final_detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            b = boxes[i]
            # Normalize to 0-1 range based on model input resolution (640x640)
            final_detections.append([
                float(b[0]) / 640.0,
                float(b[1]) / 640.0,
                float(b[2]) / 640.0,
                float(b[3]) / 640.0
            ])
            
    return final_detections

def get_aws_table(region, table_name):
    """
    Safely establishes a connection to AWS DynamoDB.
    
    Args:
        region (str): AWS region name.
        table_name (str): DynamoDB table name.
        
    Returns:
        boto3.resources.factory.Table: Table resource, or None if connection fails.
    """
    try:
        dynamodb = boto3.resource('dynamodb', region_name=region)
        table = dynamodb.Table(table_name)
        # Trigger metadata fetch to assert credentials / connection validity
        _ = table.creation_date_time
        logger.info(f"Successfully connected to AWS DynamoDB table: {table_name}")
        return table
    except Exception as e:
        logger.error(f"AWS Connection Failed: {e}")
        return None

def update_aws_table(table, data):
    """
    Sends batch updates of parking spots to AWS DynamoDB.
    
    Args:
        table: DynamoDB table instance.
        data (list): List of dict items representing spot status.
        
    Returns:
        bool: True if sync succeeded, False otherwise.
    """
    if table is None:
        logger.warning("AWS sync skipped: DynamoDB table is not connected.")
        return False
        
    try:
        with table.batch_writer() as batch:
            for item in data:
                processed_item = item.copy()
                # DynamoDB requires precise Decimal formatting for floats
                if 'Rate' in processed_item:
                    processed_item['Rate'] = Decimal(str(processed_item['Rate']))
                if 'Distance' in processed_item:
                    processed_item['Distance'] = Decimal(str(int(processed_item['Distance'])))
                batch.put_item(Item=processed_item)
        logger.info("DynamoDB update successfully batched.")
        return True
    except Exception as e:
        logger.error(f"DynamoDB Update Failed: {e}")
        return False
