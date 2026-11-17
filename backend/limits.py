"""Bound request bytes, including chunked requests, before parsing JSON/multipart."""
from starlette.responses import JSONResponse

class RequestSizeLimit:
    def __init__(self, app, max_bytes=300000):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        try:
            declared = int(headers.get(b'content-length', b'0'))
            if declared < 0:
                raise ValueError()
        except ValueError:
            return await JSONResponse({'detail':'Invalid Content-Length'}, status_code=400)(scope, receive, send)
        if declared > self.max_bytes:
            return await JSONResponse({'detail':'Request exceeds 300 KB limit.'}, status_code=413)(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > self.max_bytes:
                return await JSONResponse({'detail':'Request exceeds 300 KB limit.'}, status_code=413)(scope, receive, send)
            if not message.get('more_body', False):
                break
        replayed = False
        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {'type':'http.request', 'body':bytes(body), 'more_body':False}
            return await receive()
        await self.app(scope, replay, send)
