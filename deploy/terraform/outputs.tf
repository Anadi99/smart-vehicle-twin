output "alb_dns" {
  description = "ALB DNS name (backend API)"
  value       = aws_lb.main.dns_name
}

output "cloudfront_url" {
  description = "CloudFront URL (frontend)"
  value       = "https://${aws_cloudfront_distribution.frontend.domain_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL for backend image"
  value       = aws_ecr_repository.backend.repository_url
}

output "s3_bucket" {
  description = "S3 bucket name for frontend assets"
  value       = aws_s3_bucket.frontend.bucket
}
