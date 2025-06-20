const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5001;

const server = http.createServer((req, res) => {
    let filePath = './templates/index.html';
    if (req.url === '/') {
        filePath = path.join(__dirname, 'templates', 'index.html');
    } else {
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('<h1>404 Not Found</h1>');
        return;
    }

    fs.readFile(filePath, (err, content) => {
        if (err) {
            res.writeHead(500);
            res.end(`Server Error: ${err.code}`);
        } else {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
        console.log(`Server is running on http://127.0.0.1:${PORT}`);
    console.log('Serving index.html from ./templates/index.html');
});
