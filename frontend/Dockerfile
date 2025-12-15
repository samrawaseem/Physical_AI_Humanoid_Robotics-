# Use an official Node runtime as a parent image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --only=production

# Copy the rest of the application code
COPY . .

# Build the React app for production
RUN npm run build

# Use a lightweight web server to serve the static files
FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html

# Copy nginx configuration (optional, if you have custom config)
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 3000

# Start the server
CMD ["nginx", "-g", "daemon off;"]