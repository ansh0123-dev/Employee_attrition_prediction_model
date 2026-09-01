"""Consistent JSON response helpers."""


def success(message="Success", data=None):
    body = {"success": True, "message": message}
    if data is not None:
        body["data"] = data
    return body


def error(message="Something went wrong", errors=None):
    body = {"success": False, "message": message}
    if errors is not None:
        body["errors"] = errors
    return body
