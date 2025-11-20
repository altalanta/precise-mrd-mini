# Interactive Web Dashboard

The Precise MRD Pipeline now includes a comprehensive interactive web dashboard built with Streamlit, providing an intuitive user interface for pipeline monitoring, configuration, and result exploration.

## 🚀 Quick Start

### Run Dashboard Only
```bash
# Run dashboard with integrated API
python run_dashboard.py

# Or run dashboard directly
streamlit run src/precise_mrd/dashboard.py
```

### Run with CLI
```bash
# Start dashboard via CLI
precise-mrd dashboard

# With custom host/port
precise-mrd dashboard --host 0.0.0.0 --port 8501
```

## 🌐 Dashboard Features

### 📊 **Overview Dashboard**
- **Real-time Job Monitoring**: Live status updates for all pipeline jobs
- **Performance Metrics**: System performance and resource utilization
- **Job History**: Historical job data with filtering and search
- **Status Distribution**: Visual breakdown of job statuses

### 🚀 **Job Submission Interface**
- **Interactive Job Builder**: Point-and-click job configuration
- **Template Selection**: Pre-configured templates for common use cases
- **Real-time Validation**: Immediate feedback on configuration validity
- **Batch Job Support**: Submit multiple jobs with different parameters

### 📈 **Performance Analytics**
- **Processing Time Analysis**: Distribution and trends of job completion times
- **Resource Utilization**: CPU, memory, and storage metrics
- **Success Rate Tracking**: Historical success/failure rates
- **Comparative Analysis**: Performance comparison across different configurations

### ⚙️ **Configuration Management**
- **Visual Config Editor**: Build configurations through guided forms
- **Template Library**: Pre-built configurations for different use cases
- **Validation Tools**: Real-time configuration validation and suggestions
- **Version Control**: Configuration versioning and migration support

### 📁 **Results Explorer**
- **Interactive Data Tables**: Sortable, filterable result tables
- **Statistical Visualizations**: Histograms, scatter plots, and correlation analysis
- **Report Generation**: HTML and PDF report generation
- **Artifact Management**: Download and organize pipeline outputs

## 🔧 Technical Architecture

### Frontend (Streamlit)
```
┌─────────────────────────────────────────────────┐
│                 Streamlit App                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  Overview   │ │ Job Monitor │ │ Config Mgr  │ │
│  │ Dashboard   │ │             │ │             │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────┘
```

### Backend Integration
```
┌─────────────────────────────────────────────────┐
│              FastAPI Backend                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  Job Queue  │ │  API Routes │ │ Data Store  │ │
│  │ Management  │ │             │ │             │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────────┐
│           Pipeline Engine                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ Simulation  │ │  Processing │ │  Analysis   │ │
│  │             │ │             │ │             │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────┘
```

### Data Flow
```
User Input → Streamlit UI → API Endpoints → Pipeline Processing → Results Storage → Dashboard Display
```

## 🎛️ User Interface

### Navigation Structure
- **Overview**: Main dashboard with system status and recent activity
- **Job Monitor**: Detailed job tracking and real-time updates
- **Config Manager**: Configuration creation and validation
- **Performance**: Analytics and performance metrics
- **Documentation**: Help and API reference

### Interactive Elements
- **Real-time Updates**: Auto-refreshing data every 5 seconds
- **Filtering & Search**: Advanced filtering for job history
- **Export Options**: Download results in multiple formats
- **Responsive Design**: Optimized for desktop and mobile

## 🔌 API Integration

### REST API Endpoints Used
```python
# Job Management
POST /submit                    # Submit pipeline job
GET  /status/{job_id}          # Check job status
GET  /results/{job_id}         # Get job results
GET  /download/{job_id}/{type} # Download artifacts

# Configuration
POST /validate-config          # Validate configuration
GET  /config-templates         # Get available templates
POST /config-from-template     # Create config from template
```

### Real-time Communication
- **WebSocket Support**: Real-time updates for job progress
- **Polling Mechanism**: Fallback for environments without WebSocket
- **Event-driven Updates**: Automatic refresh on job status changes

## 📊 Visualization Components

### Charts & Graphs
- **Job Status Timeline**: Time-series view of job progression
- **Performance Histograms**: Distribution of processing times
- **Resource Utilization**: CPU, memory, and storage usage
- **Success Rate Trends**: Historical success/failure patterns

### Interactive Features
- **Zoom & Pan**: Detailed exploration of large datasets
- **Filtering**: Dynamic filtering of visualization data
- **Export**: Save visualizations as images or data files
- **Responsive Layout**: Adaptive display for different screen sizes

## 🔐 Security & Authentication

### Access Control
- **Session Management**: Secure user sessions with timeouts
- **Role-based Access**: Different permission levels for users
- **API Key Support**: Optional API key authentication

### Data Protection
- **Input Sanitization**: Comprehensive validation of all inputs
- **SQL Injection Prevention**: Parameterized queries and validation
- **XSS Protection**: Secure rendering of user-generated content

## 🚀 Deployment Options

### Local Development
```bash
# Run dashboard with API backend
python run_dashboard.py

# Run standalone dashboard
streamlit run src/precise_mrd/dashboard.py --server.port 8501
```

### Production Deployment
```bash
# Docker deployment
docker run -p 8501:8501 -p 8000:8000 mrd-dashboard

# Kubernetes deployment
kubectl apply -f k8s/dashboard-deployment.yaml
```

### Cloud Deployment
```bash
# AWS
aws ecs create-service --service-name mrd-dashboard ...

# GCP
gcloud run deploy mrd-dashboard ...

# Azure
az container create --name mrd-dashboard ...
```

## 🔧 Configuration

### Environment Variables
```bash
# Dashboard Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# API Integration
MRD_API_URL=http://localhost:8000
MRD_API_TIMEOUT=30

# Security
MRD_DASHBOARD_SECRET_KEY=your-secret-key
MRD_SESSION_TIMEOUT=3600
```

### Customization
- **Themes**: Light/dark mode support
- **Branding**: Custom logos and colors
- **Layouts**: Configurable dashboard layouts
- **Extensions**: Plugin architecture for custom components

## 📱 Mobile & Responsive Design

### Responsive Features
- **Adaptive Layouts**: Optimized for different screen sizes
- **Touch-friendly**: Large buttons and touch targets
- **Progressive Enhancement**: Core functionality works on all devices
- **Offline Support**: Cached data for offline viewing

### Mobile Optimizations
- **Simplified Navigation**: Streamlined mobile interface
- **Fast Loading**: Optimized asset loading for mobile networks
- **Data Compression**: Efficient data transfer for mobile devices

## 🔄 Real-time Updates

### Auto-refresh System
- **Configurable Intervals**: 1-30 second refresh rates
- **Smart Updates**: Only refresh changed data
- **Background Processing**: Non-blocking UI updates
- **Connection Resilience**: Automatic reconnection on network issues

### Event-driven Updates
- **WebSocket Events**: Real-time push notifications
- **Job Status Changes**: Instant updates on job completion
- **Metric Updates**: Live performance monitoring
- **Alert System**: Real-time notifications for important events

## 🎯 Use Cases

### Clinical Laboratory
- **Sample Processing**: Track patient sample analysis
- **Quality Control**: Monitor pipeline performance
- **Result Validation**: Review and approve variant calls
- **Audit Trails**: Maintain compliance records

### Research Laboratory
- **Experiment Management**: Track multiple research runs
- **Parameter Optimization**: Compare different configurations
- **Data Exploration**: Interactive result analysis
- **Collaboration**: Share results with team members

### Bioinformatician
- **Pipeline Development**: Test and debug new configurations
- **Performance Tuning**: Optimize processing parameters
- **Result Validation**: Verify analysis accuracy
- **Documentation**: Generate comprehensive reports

## 🛠️ Development

### Adding Custom Components
```python
# Create custom dashboard component
class CustomMetrics:
    def render(self):
        st.subheader("Custom Metrics")
        # Your custom visualization code here
```

### Extending Functionality
```python
# Add new dashboard page
def render_custom_page():
    st.header("Custom Analysis")
    # Custom page implementation
```

### API Integration
```python
# Custom API endpoint integration
response = requests.get(f"{API_BASE_URL}/custom/endpoint")
data = response.json()
# Process and display custom data
```

## 📚 API Reference

### Dashboard API Methods
- `st.session_state`: Persistent state management
- `st.cache_data`: Data caching for performance
- `st.rerun()`: Force dashboard refresh
- `st.experimental_rerun()`: Advanced refresh control

### Integration Patterns
- **Data Loading**: Async data fetching with caching
- **Error Handling**: Graceful error display and recovery
- **Loading States**: Progress indicators for long operations
- **State Management**: Session-based data persistence

## 🔧 Troubleshooting

### Common Issues

#### Dashboard Not Loading
- **Solution**: Check Streamlit installation and port availability
- **Debug**: Run with `--server.headless true` for headless operation

#### API Connection Issues
- **Solution**: Verify API server is running on correct port
- **Debug**: Check network connectivity and firewall settings

#### Performance Issues
- **Solution**: Enable caching and optimize data queries
- **Debug**: Monitor memory usage and processing times

#### Authentication Problems
- **Solution**: Check API key configuration and permissions
- **Debug**: Verify session management settings

## 📈 Performance Considerations

### Optimization Strategies
- **Lazy Loading**: Load data on-demand rather than upfront
- **Caching**: Use Streamlit's `@st.cache_data` decorator
- **Pagination**: Limit data display for large datasets
- **Compression**: Enable gzip compression for API responses

### Scalability Features
- **Horizontal Scaling**: Multiple dashboard instances
- **Load Balancing**: Distribute requests across instances
- **CDN Integration**: Static asset delivery optimization
- **Database Optimization**: Efficient data querying

## 🚀 Future Enhancements

### Planned Features
- **Advanced Analytics**: Machine learning-powered insights
- **Collaborative Features**: Multi-user editing and comments
- **Mobile App**: Native mobile application
- **Plugin System**: Extensible component architecture

### Integration Roadmap
- **LIMS Integration**: Direct connection to laboratory systems
- **Cloud Storage**: Native cloud provider integration
- **Advanced Security**: Multi-factor authentication
- **Audit Compliance**: Enhanced regulatory compliance features

---

**Note**: This dashboard provides a modern, user-friendly interface to the powerful MRD pipeline, making advanced bioinformatics accessible to clinicians and researchers without requiring deep technical expertise.









