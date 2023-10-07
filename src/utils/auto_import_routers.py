import importlib
import os

import loguru


def import_routers(target_path: str):
    routes = []

    for root, dirs, files in os.walk(target_path):
        for filename in files:
            if filename.endswith(".py") and filename != "__init__.py":
                root = root.replace(os.getcwd(), "")
                module_name = os.path.splitext(filename)[0]
                module_path = os.path.join(root, module_name).replace(os.sep, ".")[1:]

                try:
                    module = importlib.import_module(module_path)
                    if hasattr(module, "router"):
                        routes.append(getattr(module, "router"))
                        loguru.logger.info(f"Router imported from '{module_path}'")
                    else:
                        loguru.logger.info(f"Module '{module_path}' does not contain 'router' object")
                except ImportError:
                    loguru.logger.info(f"Failed to import module '{module_path}'")

    return routes
