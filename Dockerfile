# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Set the working directory to the Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy the application files
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the Flask app using AWS Lambda runtime
CMD ["app.lambda_handler"]
