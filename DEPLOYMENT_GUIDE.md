# 🚀 Complete Deployment Guide for Glacial Erratics Map

This guide covers multiple deployment options for your full-stack application, from simplest to most advanced.

## 📋 **What Your Project Needs to Run:**

- **Node.js 20.x** (for backend API)
- **PostgreSQL 14+** with PostGIS + pgvector extensions (for spatial data)
- **Python 3.10+ with Conda** (for spatial analysis scripts)
- **GDAL/ogr2ogr** (for processing GIS data)
- **Large GIS datasets** (~several GB of spatial data files)
- **Domain + SSL certificate** (for HTTPS)

---

## 🌟 **Option 1: Railway (EASIEST - Recommended for Beginners)**

**Cost**: ~$5-20/month • **Difficulty**: ⭐⭐☆☆☆ • **Setup Time**: 2-4 hours

### Why Railway?
- Automatic database setup with PostGIS support
- Built-in Docker container deployment
- Automatic HTTPS and domain handling
- Easy GitHub integration
- Built-in monitoring and logs

### Step-by-Step Setup:

#### **1. Prepare Your Repository**
```bash
# Commit all the new deployment files we just created
git add .
git commit -m "Add production deployment configuration"
git push origin main
```

#### **2. Set Up Railway Account**
1. Go to [railway.app](https://railway.app)
2. Sign up with your GitHub account
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your `iqp-named-erratics` repository

#### **3. Configure Database**
1. In Railway dashboard, click "Add Service" → "Database" → "PostgreSQL"
2. Wait for database to provision
3. Click on your database service → "Variables" tab
4. Note down the connection details (Railway will show them)

#### **4. Enable PostGIS Extensions**
1. In Railway dashboard, click on your PostgreSQL service
2. Go to "Query" tab
3. Run these commands:
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
```

#### **5. Configure Environment Variables**
In your main application service, go to "Variables" and add:
```
NODE_ENV=production
PORT=3001
DB_HOST=your-railway-db-host
DB_NAME=railway-db-name
DB_USER=postgres
DB_PASSWORD=your-railway-db-password
DB_PORT=5432
JWT_SECRET=your-strong-random-secret-here
```

#### **6. Upload Your GIS Data** (Critical Step)
Railway doesn't handle large file uploads well. You have two options:

**Option A: Simplified Dataset (Recommended)**
- Skip the large GIS files initially
- Your app will work but without full spatial analysis
- Users can still view erratics and use basic features

**Option B: Use External Storage**
- Upload your GIS files to AWS S3 or Google Cloud Storage
- Modify your Python scripts to download files from cloud storage
- More complex but gives full functionality

#### **7. Deploy**
1. Railway will automatically build and deploy using your Dockerfile
2. Monitor the build logs for any errors
3. Once deployed, you'll get a URL like `your-app-name.railway.app`

### **⚠️ Railway Limitations:**
- Limited storage for large GIS files
- More expensive for high CPU usage (Python analysis)
- Less control over server configuration

---

## 🔧 **Option 2: DigitalOcean App Platform (MODERATE)**

**Cost**: ~$12-25/month • **Difficulty**: ⭐⭐⭐☆☆ • **Setup Time**: 4-6 hours

### Why DigitalOcean?
- Better for larger applications
- More storage options
- Good performance/price ratio
- Managed PostgreSQL with PostGIS

### Setup Process:
1. Create DigitalOcean account
2. Create a new App from GitHub
3. Add managed PostgreSQL database
4. Configure environment variables
5. Set up file storage (Spaces) for GIS data

**Detailed steps available upon request**

---

## 🛠️ **Option 3: VPS (Virtual Private Server) - FULL CONTROL**

**Cost**: ~$10-40/month • **Difficulty**: ⭐⭐⭐⭐☆ • **Setup Time**: 6-12 hours

### Why VPS?
- Complete control over environment
- Can handle large GIS datasets
- Best performance for spatial analysis
- Most cost-effective for resource-intensive apps

### Recommended Providers:
- **DigitalOcean Droplets** ($12/month for 2GB RAM)
- **Hetzner Cloud** ($7/month for 4GB RAM - EU only)
- **Linode** ($12/month for 2GB RAM)
- **Vultr** ($12/month for 2GB RAM)

### Server Requirements:
- **Minimum**: 2GB RAM, 2 CPU cores, 50GB storage
- **Recommended**: 4GB RAM, 2 CPU cores, 100GB storage
- **For full GIS analysis**: 8GB RAM, 4 CPU cores, 200GB storage

### VPS Setup Steps:

#### **1. Create Server**
```bash
# Choose Ubuntu 22.04 LTS
# Select appropriate size based on requirements above
```

#### **2. Initial Server Setup**
```bash
# SSH into your server
ssh root@your-server-ip

# Update system
apt update && apt upgrade -y

# Install required system packages
apt install -y curl git nginx certbot python3-certbot-nginx postgresql postgresql-contrib postgis nodejs npm docker.io docker-compose

# Create application user
adduser glacial
usermod -aG sudo glacial
usermod -aG docker glacial
```

#### **3. Database Setup**
```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE glacial_erratics;
CREATE USER glacial_user WITH ENCRYPTED PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE glacial_erratics TO glacial_user;

# Connect to the new database
\c glacial_erratics

# Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
```

#### **4. Deploy Application**
```bash
# Switch to application user
su - glacial

# Clone your repository
git clone https://github.com/yourusername/iqp-named-erratics.git
cd iqp-named-erratics

# Set up environment variables
cp .env.example .env
nano .env
# Add your database credentials and other config

# Build and deploy using Docker
docker build -t glacial-erratics .
docker run -d --name glacial-app -p 3001:3001 --env-file .env glacial-erratics
```

#### **5. Set Up Nginx Reverse Proxy**
```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/glacial-erratics

# Add this configuration:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}

# Enable the site
sudo ln -s /etc/nginx/sites-available/glacial-erratics /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### **6. Set Up SSL Certificate**
```bash
# Get SSL certificate from Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

#### **7. Set Up Process Management**
```bash
# Create systemd service for your app
sudo nano /etc/systemd/system/glacial-erratics.service

# Add this content:
[Unit]
Description=Glacial Erratics App
After=network.target

[Service]
Type=forking
User=glacial
WorkingDirectory=/home/glacial/iqp-named-erratics
ExecStart=/usr/bin/docker run -d --name glacial-app -p 3001:3001 --env-file .env glacial-erratics
ExecStop=/usr/bin/docker stop glacial-app
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start the service
sudo systemctl enable glacial-erratics
sudo systemctl start glacial-erratics
```

---

## 📊 **Option Comparison**

| Feature | Railway | DigitalOcean | VPS |
|---------|---------|--------------|-----|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ |
| **Cost (monthly)** | $5-20 | $12-25 | $10-40 |
| **GIS Data Support** | Limited | Good | Excellent |
| **Performance** | Good | Very Good | Excellent |
| **Scalability** | Automatic | Good | Manual |
| **Control** | Low | Medium | High |
| **Maintenance** | None | Low | High |

---

## 🎯 **Recommended Path for You**

**Start with Railway** for these reasons:
1. **Fastest to get online** - you can have a working demo in 2-4 hours
2. **Learn the deployment process** without server management complexity
3. **Show your project to others** quickly
4. **Iterate and improve** before committing to more complex setup

**Then migrate to VPS** when you need:
- Full GIS analysis capabilities
- Better performance
- Lower long-term costs
- More control over the environment

---

## 🚨 **Critical Steps After Deployment**

### **1. Upload Initial Data**
```bash
# SSH into your server (if using VPS) or use Railway CLI
# Navigate to your project directory

# Run database migrations
npm run db:migrate

# Import initial erratic data
cd backend && npm run db:import

# Run spatial analysis (this will take time!)
cd src/scripts/python
conda activate glacial-erratics
python run_analysis.py --all
```

### **2. Set Up Your GIS Data Files**
You need to manually upload these large files to your server:
- Native Land territories, languages, treaties GeoJSON files
- North American Roads Database (NATD) shapefiles
- National Forest System Trails shapefiles
- GMTED2010 DEM tiles
- OpenStreetMap PBF files

### **3. Configure Domain and SSL**
- Point your domain to your server IP
- Set up SSL certificate for HTTPS
- Update your environment variables with the new domain

### **4. Set Up Monitoring**
- Configure log monitoring
- Set up uptime monitoring
- Monitor database performance
- Track Python script execution times

---

## 🆘 **Need Help?**

**If you get stuck:**
1. Check the application logs first
2. Verify all environment variables are set correctly
3. Ensure database extensions are installed
4. Check that conda environment is activated
5. Verify GIS data files are in the correct locations

**Common Issues:**
- **Python scripts fail**: Conda environment not activated
- **Database connection errors**: Check credentials and network access
- **Frontend not loading**: Build files not generated or served correctly
- **Analysis endpoints timeout**: GIS data files missing or inaccessible

Would you like me to walk you through any specific deployment option in detail? 