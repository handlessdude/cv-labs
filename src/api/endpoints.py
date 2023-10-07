import os

import fastapi

from src.api.routes import api_path
from src.utils.auto_import_routers import import_routers

api_endpoint_router = fastapi.APIRouter()

routes = import_routers(api_path)
for router in routes:
    api_endpoint_router.include_router(router=router)
