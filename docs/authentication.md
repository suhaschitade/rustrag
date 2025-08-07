# Authentication & Rate Limiting

This document describes the authentication and rate limiting system implemented in Activity 5.5.

## Overview

The RustRAG API implements a comprehensive authentication and rate limiting system with the following features:

- **API Key-based Authentication**: Secure API key management with permissions and expiration
- **Permission-based Authorization**: Granular permission system (Read, Write, Delete, Admin, All)
- **Rate Limiting**: Per-user and per-IP rate limiting with customizable limits
- **Admin Management**: Complete CRUD operations for API key management
- **Development Features**: Permission testing endpoints and monitoring

## Authentication System

### API Key Management

The system uses API keys for authentication. Each API key has:

- **Unique ID**: UUID identifier
- **Name & Description**: Human-readable labels
- **Permissions**: Granular permission system
- **Rate Limits**: Custom rate limiting per key
- **Expiration**: Optional expiration dates
- **Usage Tracking**: Last used time and usage count
- **Status**: Active/inactive state

### Permission Levels

```rust
pub enum Permission {
    Read,    // Can read documents and query
    Write,   // Can upload and modify documents  
    Delete,  // Can delete documents
    Admin,   // Can perform administrative operations
    All,     // Can access all operations (super admin)
}
```

### Default Admin Key

The system automatically creates a default admin API key on startup:
- **Format**: `rag_[UUID without hyphens]`
- **Permissions**: `All` (super admin)
- **Rate Limit**: None (unlimited)
- **Expiration**: Never expires

**⚠️ Important**: Replace the default admin key in production!

## API Endpoints

### Authentication Management

#### Create API Key
```http
POST /api/v1/auth/keys
Authorization: Bearer <admin_api_key>
Content-Type: application/json

{
    "name": "My API Key",
    "description": "Key for my application",
    "permissions": ["Read", "Write"],
    "rate_limit_per_hour": 500,
    "expires_in_days": 90
}
```

#### List API Keys (Admin Only)
```http
GET /api/v1/auth/keys?page=1&limit=10
Authorization: Bearer <admin_api_key>
```

#### Get Specific API Key
```http
GET /api/v1/auth/keys/{key_id}
Authorization: Bearer <api_key>
```

#### Update API Key (Admin Only)
```http
PATCH /api/v1/auth/keys/{key_id}
Authorization: Bearer <admin_api_key>
Content-Type: application/json

{
    "name": "Updated Name",
    "permissions": ["Read", "Write", "Delete"],
    "rate_limit_per_hour": 1000,
    "is_active": true
}
```

#### Revoke API Key (Admin Only)
```http
POST /api/v1/auth/keys/{key_id}/revoke
Authorization: Bearer <admin_api_key>
```

#### Delete API Key (Admin Only)
```http
DELETE /api/v1/auth/keys/{key_id}
Authorization: Bearer <admin_api_key>
```

### Authentication Status

#### Get Current User Info
```http
GET /api/v1/auth/me
Authorization: Bearer <api_key>
```

#### Get Rate Limit Status
```http
GET /api/v1/auth/rate-limit
Authorization: Bearer <api_key>
```

#### Get All Rate Limit Statuses (Admin Only)
```http
GET /api/v1/auth/rate-limits
Authorization: Bearer <admin_api_key>
```

#### Get Authentication Statistics (Admin Only)
```http
GET /api/v1/auth/stats
Authorization: Bearer <admin_api_key>
```

### Permission Testing (Development)

```http
GET /api/v1/auth/test/read     # Test Read permission
GET /api/v1/auth/test/write    # Test Write permission  
GET /api/v1/auth/test/delete   # Test Delete permission
GET /api/v1/auth/test/admin    # Test Admin permission
```

## Rate Limiting System

### Features

- **Multiple Strategies**: Per-IP and per-API-key rate limiting
- **Sliding Window**: Time-based sliding window algorithm
- **Custom Limits**: Per-API-key custom rate limits
- **Headers**: Standard rate limit headers in responses
- **Monitoring**: Real-time rate limit status monitoring

### Rate Limit Headers

All responses include rate limiting headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1673023200
```

When rate limit is exceeded:
```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1673023200
Retry-After: 3600

{
    "error": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 3600 seconds",
    "code": 429
}
```

### Configuration

Rate limiting can be configured via environment variables:

```bash
# Enable/disable rate limiting
RATE_LIMITING_ENABLED=true

# Default limits (can be overridden per API key)
RATE_LIMIT_UNAUTHENTICATED_RPM=100    # Per IP
RATE_LIMIT_AUTHENTICATED_RPM=1000     # Per API key
```

## Usage Examples

### Basic Authentication

```bash
# Using X-API-Key header
curl -H "X-API-Key: rag_1234567890abcdef" \
     https://api.example.com/api/v1/documents

# Using Authorization Bearer token
curl -H "Authorization: Bearer rag_1234567890abcdef" \
     https://api.example.com/api/v1/documents
```

### Creating an API Key

```bash
# Create a new API key (requires admin permissions)
curl -X POST \
  -H "Authorization: Bearer <admin_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production API Key",
    "description": "API key for production application",
    "permissions": ["Read", "Write"],
    "rate_limit_per_hour": 2000,
    "expires_in_days": 365
  }' \
  https://api.example.com/api/v1/auth/keys
```

### JavaScript Example

```javascript
const API_KEY = 'rag_1234567890abcdef';
const BASE_URL = 'https://api.example.com/api/v1';

// Configure default headers
const headers = {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
};

// Upload a document
async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', file.name);
    
    const response = await fetch(`${BASE_URL}/documents`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${API_KEY}`
        },
        body: formData
    });
    
    // Check rate limit headers
    console.log('Rate limit remaining:', response.headers.get('X-RateLimit-Remaining'));
    
    return response.json();
}

// Query documents  
async function queryDocuments(query) {
    const response = await fetch(`${BASE_URL}/query`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            query: query,
            max_results: 10
        })
    });
    
    return response.json();
}
```

### Python Example

```python
import requests
import time

class RustRAGClient:
    def __init__(self, api_key, base_url="https://api.example.com/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def upload_document(self, file_path, title=None):
        """Upload a document"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'title': title or file_path}
            
            # Remove Content-Type header for multipart upload
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            response = requests.post(
                f'{self.base_url}/documents',
                headers=headers,
                files=files,
                data=data
            )
            
            # Check rate limiting
            self._check_rate_limit(response)
            
            return response.json()
    
    def query_documents(self, query, max_results=10):
        """Query documents"""
        response = self.session.post(
            f'{self.base_url}/query',
            json={
                'query': query,
                'max_results': max_results
            }
        )
        
        self._check_rate_limit(response)
        return response.json()
    
    def get_rate_limit_status(self):
        """Get current rate limit status"""
        response = self.session.get(f'{self.base_url}/auth/rate-limit')
        return response.json()
    
    def _check_rate_limit(self, response):
        """Check and handle rate limiting"""
        remaining = response.headers.get('X-RateLimit-Remaining')
        if remaining and int(remaining) < 10:
            print(f"Warning: Only {remaining} requests remaining")
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            raise Exception("Rate limit exceeded")

# Usage
client = RustRAGClient('rag_1234567890abcdef')

# Upload a document
result = client.upload_document('./document.pdf', 'Important Document')
print(f"Document uploaded: {result['data']['id']}")

# Query documents
results = client.query_documents('What is the main topic?')
print(f"Found {len(results['data'])} results")

# Check rate limit status
status = client.get_rate_limit_status()
print(f"Rate limit: {status['data']['remaining']}/{status['data']['limit']}")
```

## Security Considerations

### Production Deployment

1. **Replace Default Admin Key**: The default admin key should be replaced in production
2. **Use HTTPS**: Always use HTTPS in production
3. **Rotate Keys Regularly**: Implement key rotation policies
4. **Monitor Usage**: Monitor API key usage for suspicious activity
5. **Set Expiration**: Use key expiration for enhanced security

### Best Practices

1. **Principle of Least Privilege**: Grant minimal required permissions
2. **Environment Variables**: Store API keys in environment variables, not code
3. **Key Management**: Use dedicated key management systems for production
4. **Logging**: Monitor authentication failures and rate limit violations
5. **Backup Admin Access**: Maintain multiple admin keys for emergency access

### Rate Limiting Best Practices

1. **Gradual Backoff**: Implement exponential backoff for rate limited requests
2. **Monitor Headers**: Always check rate limit headers in responses  
3. **Batch Operations**: Use batch endpoints when available
4. **Cache Results**: Cache results to reduce API calls
5. **Handle 429 Errors**: Implement proper 429 error handling

## Monitoring & Troubleshooting

### Logs

The system logs authentication events:

```
INFO Authentication successful for API key: Default Admin Key (12345...)
WARN Authentication failed: Invalid API key
INFO Rate limit check passed for client: api_key:12345... (999/1000 remaining)
WARN Rate limit exceeded for client: ip:192.168.1.100 (0/100 remaining)
```

### Common Issues

1. **401 Unauthorized**: Missing or invalid API key
2. **403 Forbidden**: Insufficient permissions for the operation
3. **429 Too Many Requests**: Rate limit exceeded
4. **API key not working**: Check if key is active and not expired

### Health Checks

Monitor authentication system health:

```bash
# Check system stats
curl -H "Authorization: Bearer <admin_key>" \
     https://api.example.com/api/v1/auth/stats

# Check rate limit status
curl -H "Authorization: Bearer <admin_key>" \
     https://api.example.com/api/v1/auth/rate-limits
```

## Future Enhancements

Planned improvements for future releases:

1. **Database Storage**: Move from in-memory to database storage
2. **Redis Integration**: Use Redis for rate limiting and caching  
3. **JWT Support**: Add JWT token support alongside API keys
4. **RBAC**: Role-based access control system
5. **OAuth2**: OAuth2/OpenID Connect integration
6. **Audit Logging**: Comprehensive audit trail
7. **Multi-tenancy**: Support for multiple tenants/organizations
8. **Key Scoping**: Resource-specific key scoping
9. **Webhook Integration**: Webhooks for key events
10. **Advanced Analytics**: Detailed usage analytics and reporting
