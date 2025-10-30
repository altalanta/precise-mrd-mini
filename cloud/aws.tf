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

# ECS Cluster
resource "aws_ecs_cluster" "mrd_cluster" {
  name = "precise-mrd-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Environment = var.environment
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "mrd_task" {
  family                   = "precise-mrd-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "mrd-api"
      image = "precise-mrd-api:latest"

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "PRECISE_MRD_LOG_LEVEL"
          value = "INFO"
        },
        {
          name  = "ENABLE_PARALLEL_PROCESSING"
          value = "true"
        },
        {
          name  = "ENABLE_ML_CALLING"
          value = "true"
        },
        {
          name  = "ENABLE_DEEP_LEARNING"
          value = "true"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/precise-mrd-api"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      healthCheck = {
        command = [
          "CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"
        ]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Environment = var.environment
  }
}

# ECS Service
resource "aws_ecs_service" "mrd_service" {
  name            = "precise-mrd-api-service"
  cluster         = aws_ecs_cluster.mrd_cluster.id
  task_definition = aws_ecs_task_definition.mrd_task.arn
  desired_count   = 3

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 100
  }

  network_configuration {
    subnets          = [aws_subnet.mrd_subnet.id]
    security_groups  = [aws_security_group.mrd_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.mrd_target_group.arn
    container_name   = "mrd-api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.mrd_listener]

  tags = {
    Environment = var.environment
  }
}

# Load Balancer
resource "aws_lb" "mrd_lb" {
  name               = "precise-mrd-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.mrd_sg.id]
  subnets            = [aws_subnet.mrd_subnet.id]

  enable_deletion_protection = false

  tags = {
    Environment = var.environment
  }
}

resource "aws_lb_target_group" "mrd_target_group" {
  name        = "precise-mrd-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.mrd_vpc.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }

  tags = {
    Environment = var.environment
  }
}

resource "aws_lb_listener" "mrd_listener" {
  load_balancer_arn = aws_lb.mrd_lb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mrd_target_group.arn
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
  value       = "http://${aws_lb.mrd_lb.dns_name}"
}






