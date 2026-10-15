"""HTTP helpers. Bodies stay FastAPI-shaped: ``{"detail": ...}``. Never include secrets."""

from fastapi import HTTPException, status


def not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


def unavailable(detail: str) -> HTTPException:
    """Optional 503 when a dependency is down; historical queries should still be attempted."""
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def not_implemented(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=detail)
