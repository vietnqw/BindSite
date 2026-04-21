import logging
import sys
from typing import Optional

# Change WARNING level name to WARN to match requested format
logging.addLevelName(logging.WARNING, 'WARN')

class ColorFormatter(logging.Formatter):
    """Custom color formatter with ANSI escape codes"""
    
    # ANSI Colors
    TIME_COLOR = "\x1b[38;5;243m"  # Gray
    SEP_COLOR = "\x1b[38;5;238m"   # Darker Gray
    
    LEVEL_COLORS = {
        logging.DEBUG: "\x1b[38;5;39m",     # Blue
        logging.INFO: "\x1b[38;5;46m",      # Green
        logging.WARNING: "\x1b[38;5;214m",  # Orange/Yellow
        logging.ERROR: "\x1b[38;5;196m",    # Red
        logging.CRITICAL: "\x1b[1m\x1b[38;5;196m", # Bold Red
    }
    
    # Colors for the message itself, varying by log level
    MSG_COLORS = {
        logging.DEBUG: "\x1b[38;5;250m",    # Light Gray
        logging.INFO: "\x1b[0m",            # Default
        logging.WARNING: "\x1b[38;5;214m",  # Orange/Yellow
        logging.ERROR: "\x1b[38;5;196m",    # Red
        logging.CRITICAL: "\x1b[1m\x1b[38;5;196m", # Bold Red
    }
    RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and specific layout"""
        message = record.getMessage()
        
        # If 'simple=True' is passed in extra, just return the message
        if getattr(record, "simple", False):
            msg_color = self.MSG_COLORS.get(record.levelno, self.RESET)
            return f"{msg_color}{message}{self.RESET}"

        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level_name = f"{record.levelname:<5}"
        
        level_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        msg_color = self.MSG_COLORS.get(record.levelno, self.RESET)
        
        sep = f"{self.SEP_COLOR} | {self.RESET}"
        
        formatted = (
            f"{self.TIME_COLOR}{time_str}{self.RESET}"
            f"{sep}"
            f"{level_color}{level_name}{self.RESET}"
            f"{sep}"
            f"{msg_color}{message}{self.RESET}"
        )
        return formatted




def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup and return a logger with beautiful colored formatting.
    
    Args:
        name: The name of the logger (e.g. 'data_loader', 'trainer').
        level: Default logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, assume it's configured and return it
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    # Prevent propagation to avoid duplicate logs if root logger is also configured
    logger.propagate = False
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = ColorFormatter()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger
