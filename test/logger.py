import logging


logger = None

def get_logger(name: str="summarizer-map-reduce-test") -> logging.Logger:
    """
    Get a logger with the specified name. If no handlers are set, it will create a default StreamHandler.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    global logger

    if logger:
        # If the logger is already created, return it
        return logger

    # Validate the logger name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Logger name must be a non-empty string.")

    # Get or create a logger with the specified name
    logger = logging.getLogger(name)

    # Ensure the logger is not already configured
    if not logger.hasHandlers():
        # If the logger does not have handlers, we will set it up
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)

    return logger