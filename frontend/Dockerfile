# Build the React/Vite frontend
FROM node:20-alpine AS build

WORKDIR /frontend

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Serve the production build with nginx
FROM nginx:alpine

COPY --from=build /frontend/dist /usr/share/nginx/html

# SPA fallback so React routes resolve correctly.
RUN printf '%s\n' \
    'server {' \
    '    listen 80;' \
    '    server_name _;' \
    '    root /usr/share/nginx/html;' \
    '    index index.html;' \
    '    location / {' \
    '        try_files $uri $uri/ /index.html;' \
    '    }' \
    '}' \
    > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
