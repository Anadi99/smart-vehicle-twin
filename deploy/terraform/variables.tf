variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-west-1"
}

variable "project" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "svt"
}
