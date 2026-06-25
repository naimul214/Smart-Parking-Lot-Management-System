import json
import boto3
import os
from decimal import Decimal

# Helper class to convert DynamoDB Decimal to Python float/int for JSON serialization
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            if obj % 1 == 0:
                return int(obj)
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

# Initialize DynamoDB Table resource
DYNAMO_TABLE_NAME = os.environ.get("DYNAMO_TABLE_NAME", "ParkingData")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DYNAMO_TABLE_NAME)

def lambda_handler(event, context):
    """
    AWS Lambda handler for retrieving current parking spot statuses.
    Triggered by HTTP GET requests through API Gateway.
    
    Args:
        event (dict): API Gateway event containing GET query parameters.
        context: Lambda execution context.
        
    Returns:
        dict: HTTP response status containing parking lot status.
    """
    try:
        # Scan the table (appropriate for small datasets like a single parking lot)
        # For production large-scale systems, use query on secondary indexes
        response = table.scan()
        items = response.get("Items", [])
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(items, cls=DecimalEncoder)
        }
        
    except Exception as e:
        print(f"Error executing Lambda GET: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Internal database error: {str(e)}"})
        }
