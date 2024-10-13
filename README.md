Image Classification API with AWS Integration
This project provides an image classification API built using FastAPI and integrates with AWS S3 for storing images and logs. It loads a pre-trained TensorFlow model from S3 to classify images into different categories. The API supports uploading images for predictions, as well as performing batch predictions on images stored in an S3 bucket.

Features
Predict single image: Allows users to upload an image for classification via the API.
Batch prediction: Processes multiple images stored in an S3 bucket and returns predictions for each.
AWS S3 Integration: Downloads the TensorFlow model, processes images, and uploads logs back to S3.
Logging: Logs the operations (model loading, predictions) and stores them in an S3 bucket.

Getting Started
These instructions will help you set up and run the project on your local machine or via GitHub Actions.

Prerequisites:
Python
TensorFlow
FastAPI
Boto3 (AWS SDK for Python)
AWS S3 bucket with required permissions
AWS Credentials (Access key and secret key)

AWS Setup
Before running the application, ensure you have the following:

An AWS S3 bucket for storing images, logs, and the trained model.
AWS Credentials properly set up (you can use the AWS CLI or environment variables).
Upload your pre-trained model to the S3 bucket (Image_resnet50.h5).
Set up a folder in the S3 bucket (predict/) to store images for batch prediction.

GitHub Actions
This project includes a GitHub Actions workflow to automate batch predictions at specific times (e.g., 8:35 PM daily).

Setting up GitHub Actions
Ensure your AWS credentials are set as secrets in your GitHub repository:

AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
The GitHub Actions workflow runs the model and performs batch predictions from S3. To modify the schedule (default is 8:35 PM South African time), you can edit the cron schedule in the yaml file.

Deployment
To deploy this API, you can either:

Run it locally using uvicorn.
Set it up on a cloud platform (AWS EC2, Heroku, etc.) with your environment set up for FastAPI and AWS integration.

Author
- Stephen Moorcroft

Copyright (c) 2024 Stephen Moorcroft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, and/or sublicense. make a read me files like this for this task, anybody can use the code freely.
