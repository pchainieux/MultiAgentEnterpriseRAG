from __future__ import annotations

from fastapi import Depends

from src.graph.workflow import get_graph_app
from src.app.core.logging import get_logger


def get_logger_dep():
    """
    Provide a request scoped logger.
    """
    return get_logger("api")


def get_graph_app_dep():
    """
    Provide compiled LangGraph app for dependency injection in routes.
    """
    return get_graph_app()


GraphApp = type(get_graph_app()) 


def common_dependencies(
    logger=Depends(get_logger_dep),
    graph_app=Depends(get_graph_app_dep),
):
    """
    Aggregates common dependencies.
    """
    return {"logger": logger, "graph_app": graph_app}
