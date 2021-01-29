FROM node:alpine

RUN apk add --no-cache --virtual .gyp python make g++ 

WORKDIR /app

# We copy just the dependencies first to leverage Docker cache
COPY package.json /app
COPY yarn.lock /app

RUN yarn install

COPY . /app

CMD ["yarn", "start"]
