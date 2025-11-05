# AWS Terraform configuration for Precise MRD Pipeline

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

# Provider configuration
provider "aws" {
  region = var.aws_region
}

# VPC and networking
resource "aws_vpc" "mrd_vpc" {
  cidr_block = var.vpc_cidr
  tags = {
    Name = "precise-mrd-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "mrd_subnet" {
  vpc_id     = aws_vpc.mrd_vpc.id
  cidr_block = "10.0.1.0/24"
  availability_zone = "${var.aws_region}a"
  tags = {
    Name = "precise-mrd-subnet"
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "mrd_igw" {
  vpc_id = aws_vpc.mrd_vpc.id
  tags = {
    Name = "precise-mrd-igw"
    Environment = var.environment
  }
}

resource "aws_route_table" "mrd_route_table" {
  vpc_id = aws_vpc.mrd_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mrd_igw.id
  }

  tags = {
    Name = "precise-mrd-route-table"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "mrd_rta" {
  subnet_id      = aws_subnet.mrd_subnet.id
  route_table_id = aws_route_table.mrd_route_table.id
}

# Security groups
resource "aws_security_group" "mrd_sg" {
  name        = "precise-mrd-sg"
  description = "Security group for Precise MRD API"
  vpc_id      = aws_vpc.mrd_vpc.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "API Port"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "precise-mrd-sg"
    Environment = var.environment
  }
}

# IAM Roles
resource "aws_iam_role" "ecs_execution_role" {
  name = "precise-mrd-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
  ]
}

resource "aws_iam_role" "ecs_task_role" {
  name = "precise-mrd-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  inline_policy {
    name = "mrd-task-policy"
    policy = jsonencode({
      Version = "2012-10-17"
      Statement = [
        {
          Effect = "Allow"
          Action = [
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
          ]
          Resource = "arn:aws:logs:*:*:*"
        }
      ]
    })
  }
}

# ECR Repository
resource "aws_ecr_repository" "mrd_ecr" {
  name                 = "precise-mrd-pipeline"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
  }
}

# IAM Roles for AWS Batch
resource "aws_iam_role" "batch_service_role" {
  name = "precise-mrd-batch-service-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Action    = "sts:AssumeRole",
      Effect    = "Allow",
      Principal = { Service = "batch.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service_role_attachment" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role" "ecs_instance_role" {
  name = "precise-mrd-ecs-instance-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Action    = "sts:AssumeRole",
      Effect    = "Allow",
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_instance_role_attachment" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ecs_instance_profile" {
  name = "precise-mrd-ecs-instance-profile"
  role = aws_iam_role.ecs_instance_role.name
}

resource "aws_iam_role" "batch_job_role" {
  name = "precise-mrd-batch-job-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Action    = "sts:AssumeRole",
      Effect    = "Allow",
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "batch_job_policy" {
  name = "precise-mrd-batch-job-policy"
  role = aws_iam_role.batch_job_role.id
  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      Resource = "*" # Restrict in production
    }]
  })
}

# AWS Batch Compute Environment
resource "aws_batch_compute_environment" "mrd_batch_ce" {
  compute_environment_name = "precise-mrd-compute-env"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service_role.arn

  compute_resources {
    type                      = "FARGATE"
    max_vcpus                 = 16
    subnets                   = [aws_subnet.mrd_subnet.id]
    security_group_ids        = [aws_security_group.mrd_sg.id]
  }

  tags = {
    Environment = var.environment
  }
}

# AWS Batch Job Queue
resource "aws_batch_job_queue" "mrd_job_queue" {
  name     = "precise-mrd-job-queue"
  state    = "ENABLED"
  priority = 1

  compute_environments = [
    aws_batch_compute_environment.mrd_batch_ce.arn
  ]

  tags = {
    Environment = var.environment
  }
}

# AWS Batch Job Definition
resource "aws_batch_job_definition" "mrd_job_def" {
  name = "precise-mrd-job-definition"
  type = "container"

  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image       = "${aws_ecr_repository.mrd_ecr.repository_url}:latest",
    jobRoleArn  = aws_iam_role.batch_job_role.arn,
    executionRoleArn = aws_iam_role.ecs_execution_role.arn,
    resourceRequirements = [
      { type = "VCPU", value = "2" },
      { type = "MEMORY", value = "4096" }
    ],
    command = [
      "Ref::command"
    ],
    environment = [
      { name = "PRECISE_MRD_LOG_LEVEL", value = "INFO" }
    ]
  })

  tags = {
    Environment = var.environment
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "mrd_api" {
  name          = "precise-mrd-api"
  protocol_type = "HTTP"
  target        = aws_lambda_function.job_submitter.invoke_arn

  tags = {
    Environment = var.environment
  }
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id           = aws_apigatewayv2_api.mrd_api.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.job_submitter.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "submit_route" {
  api_id    = aws_apigatewayv2_api.mrd_api.id
  route_key = "POST /submit"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "api_stage" {
  api_id      = aws_apigatewayv2_api.mrd_api.id
  name        = "$default"
  auto_deploy = true
}

# Lambda Function
resource "aws_lambda_function" "job_submitter" {
  function_name = "precise-mrd-job-submitter"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.11"
  filename      = "../serverless/lambda_function.zip" # Placeholder

  environment {
    variables = {
      JOB_QUEUE_ARN = aws_batch_job_queue.mrd_job_queue.arn
      JOB_DEFINITION_ARN = aws_batch_job_definition.mrd_job_def.arn
    }
  }

  tags = {
    Environment = var.environment
  }
}

resource "aws_iam_role" "lambda_exec_role" {
  name = "precise-mrd-lambda-exec-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Action    = "sts:AssumeRole",
      Effect    = "Allow",
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "precise-mrd-lambda-policy"
  role = aws_iam_role.lambda_exec_role.id
  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "batch:SubmitJob",
        Resource = "*"
      },
      {
        Effect   = "Allow",
        Action   = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_lambda_permission" "api_gateway_permission" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.job_submitter.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.mrd_api.execution_arn}/*/*"
}
# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "mrd_logs" {
  name              = "/ecs/precise-mrd-api"
  retention_in_days = 30

  tags = {
    Environment = var.environment
  }
}

# Outputs
output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.mrd_lb.dns_name
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = aws_apigatewayv2_api.mrd_api.api_endpoint
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.mrd_ecr.repository_url
}

output "batch_job_queue_arn" {
  description = "ARN of the AWS Batch Job Queue"
  value       = aws_batch_job_queue.mrd_job_queue.arn
}

output "batch_job_definition_arn" {
  description = "ARN of the AWS Batch Job Definition"
  value       = aws_batch_job_definition.mrd_job_def.arn
}


