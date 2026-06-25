import json
import boto3
import os
from decimal import Decimal

# Initialize DynamoDB Table resource
DYNAMO_TABLE_NAME = os.environ.get("DYNAMO_TABLE_NAME", "ParkingData")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DYNAMO_TABLE_NAME)

def lambda_handler(event, context):
    """
    AWS Lambda handler for updating parking spot statuses.
    Triggered by HTTP POST requests through API Gateway.
    
    Args:
        event (dict): API Gateway event containing POST request body.
        context: Lambda execution context.
        
    Returns:
        dict: HTTP response status and message.
    """
    try:
        # Check if the body contains data
        body = event.get("body", "[]")
        if isinstance(body, str):
            data = json.loads(body)
        else:
            data = body
            
        if not data or not isinstance(data, list):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Invalid request payload. Expected a list of parking spots."})
            }
            
        # Write batch items into DynamoDB
        with table.batch_writer() as batch:
            for item in data:
                # Format to DynamoDB decimal compatibility
                db_item = {
                    "SpotID": str(item.get("SpotID")),
                    "Status": str(item.get("Status")),
                    "Distance": Decimal(str(item.get("Distance", 0))),
                    "Rate": Decimal(str(item.get("Rate", 0.05)))
                }
                batch.put_item(Item=db_item)
                
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"status": "success", "message": f"Updated {len(data)} parking spots."})
        }
        
    except Exception as e:
        print(f"Error executing Lambda POST: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Internal server error: {str(e)}"})
        }
